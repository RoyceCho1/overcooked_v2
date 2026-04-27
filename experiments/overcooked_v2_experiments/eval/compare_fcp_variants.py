import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
from tqdm import tqdm

import jaxmarl
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper
from jaxmarl.environments.overcooked_v2.common import DynamicObject

from overcooked_v2_experiments.ppo.policy import PPOPolicy
from overcooked_v2_experiments.ppo.utils.store import load_checkpoint
from overcooked_v2_experiments.recipe.context import RecipeContextManager
from overcooked_v2_experiments.recipe.masking import get_mask_fn
from overcooked_v2_experiments.recipe.recipe_encoder_jax import RecipeEncoder

DETAIL_COLUMNS = [
    "variant",
    "pair_order",
    "fcp_run",
    "partner_run",
    "episodes",
    "max_steps",
    "seed",
    "mean_return",
    "std_return",
]


@dataclass
class VariantEvalConfig:
    name: str
    fcp_dir: Path
    context_mode: str  # "none", "encoder", "oracle"


class ContextualPPOPolicy(PPOPolicy):
    """PPO policy wrapper that supports optional recipe context input."""

    def compute_action(self, obs, done, hstate, key, context=None, params=None):
        if context is None:
            return super().compute_action(obs, done, hstate, key, params=params)

        if params is None:
            params = self.params
        assert params is not None

        done = jnp.array(done)

        def _add_dim(tree):
            return jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], tree)

        ac_in = (obs, done, context)
        ac_in = _add_dim(ac_in)
        if not self.with_batching:
            ac_in = _add_dim(ac_in)

        next_hstate, pi, _ = self.network.apply(params, hstate, ac_in)

        if self.stochastic:
            action = pi.sample(seed=key)
        else:
            action = jnp.argmax(pi.probs, axis=-1)

        if self.with_batching:
            action = action[0]
        else:
            action = action[0, 0]

        return action, next_hstate


def _list_run_ids(run_dir: Path) -> List[int]:
    run_ids = []
    for p in run_dir.iterdir():
        if not p.is_dir() or not p.name.startswith("run_"):
            continue
        try:
            run_id = int(p.name.split("_")[1])
        except Exception:
            continue
        if (p / "ckpt_final").exists():
            run_ids.append(run_id)
    return sorted(run_ids)


def _load_policy(run_dir: Path, run_id: int, stochastic: bool = False) -> ContextualPPOPolicy:
    config, params = load_checkpoint(run_dir, run_id, "final")
    return ContextualPPOPolicy(
        params=params,
        config=config,
        stochastic=stochastic,
        with_batching=True,
    )


def _build_recipe_codebook(env) -> np.ndarray:
    recipes = jnp.array(env.layout.possible_recipes, dtype=jnp.int32)
    recipe_codes = jax.vmap(DynamicObject.get_recipe_encoding)(recipes)
    return np.array(recipe_codes, dtype=np.int32)


def _recipe_bits_from_state(state):
    # OvercookedV2LogWrapper keeps env state under `env_state`.
    if hasattr(state, "env_state"):
        return state.env_state.recipe
    return state.recipe


def _recipe_label(recipe_bits, recipe_codes):
    """Map a scalar raw recipe bit-encoding to class id without host sync."""
    matches = recipe_codes == recipe_bits
    has_match = jnp.any(matches)
    label = jnp.argmax(matches.astype(jnp.int32)).astype(jnp.int32)
    return jnp.where(has_match, label, jnp.int32(0))


def _recipe_labels(recipe_bits, recipe_codes):
    """Vectorized recipe-bit -> class-id mapper."""
    recipe_bits = jnp.asarray(recipe_bits)
    if recipe_bits.ndim == 0:
        return _recipe_label(recipe_bits, recipe_codes)
    return jax.vmap(lambda bits: _recipe_label(bits, recipe_codes))(recipe_bits)


def _load_encoder(
    encoder_ckpt: Path,
    obs_shape: Tuple[int, ...],
    num_classes: int,
    k: int,
    use_actions: bool,
    action_dim: int,
):
    model = RecipeEncoder(
        num_actions=action_dim if use_actions else 0,
        num_classes=num_classes,
        use_actions=use_actions,
    )

    dummy_obs = jnp.zeros((1, k, *obs_shape), dtype=jnp.float32)
    if use_actions:
        dummy_act = jnp.zeros((1, k, action_dim), dtype=jnp.float32)
        dummy_params = model.init(jax.random.PRNGKey(0), dummy_obs, dummy_act)["params"]
    else:
        dummy_params = model.init(jax.random.PRNGKey(0), dummy_obs)["params"]

    checkpointer = ocp.PyTreeCheckpointer()
    params = checkpointer.restore(str(encoder_ckpt), item=dummy_params)
    return model, params


def _load_json_if_exists(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_encoder_k(encoder_meta_path: Path, cli_encoder_k: Optional[int]) -> int:
    if cli_encoder_k is not None:
        return int(cli_encoder_k)

    encoder_meta = _load_json_if_exists(encoder_meta_path)
    if "k" in encoder_meta:
        return int(encoder_meta["k"])
    if "segment_k" in encoder_meta:
        return int(encoder_meta["segment_k"])

    data_dir = encoder_meta.get("data_dir", None)
    if data_dir:
        dataset_meta_path = Path(data_dir) / "dataset_meta.json"
        dataset_meta = _load_json_if_exists(dataset_meta_path)
        if "segment_k" in dataset_meta:
            return int(dataset_meta["segment_k"])

    return 10


def _load_existing_detail(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=DETAIL_COLUMNS)

    for col in DETAIL_COLUMNS:
        if col not in df.columns:
            if col == "pair_order":
                df[col] = -1
            else:
                df[col] = np.nan
    return df[DETAIL_COLUMNS]


def _append_detail_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists()) or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, None) for k in DETAIL_COLUMNS})
        f.flush()
        os.fsync(f.fileno())


def _episode_key(seed: int, fcp_run_id: int, partner_run_id: int, ep_idx: int):
    key = jax.random.PRNGKey(seed)
    key = jax.random.fold_in(key, jnp.asarray(fcp_run_id, dtype=jnp.uint32))
    key = jax.random.fold_in(key, jnp.asarray(partner_run_id, dtype=jnp.uint32))
    key = jax.random.fold_in(key, jnp.asarray(ep_idx, dtype=jnp.uint32))
    return key


def _make_scanned_pair_runner(
    env_reset_vmapped,
    env_step_vmapped,
    fcp_policy: ContextualPPOPolicy,
    partner_policy: ContextualPPOPolicy,
    variant: VariantEvalConfig,
    context_manager: Optional[RecipeContextManager],
    num_episodes: int,
    episode_limit: int,
    stochastic: bool,
) -> Callable:
    """Build a jitted pair evaluator that scans timesteps on-device."""

    context_mode = variant.context_mode

    def _policy_action(policy, params, obs, done, hstate, key, context=None):
        done = jnp.asarray(done)
        if context is None:
            ac_in = (obs[jnp.newaxis, ...], done[jnp.newaxis, ...])
        else:
            ac_in = (
                obs[jnp.newaxis, ...],
                done[jnp.newaxis, ...],
                context[jnp.newaxis, ...],
            )

        next_hstate, pi, _ = policy.network.apply(params, hstate, ac_in)
        if stochastic:
            action = pi.sample(seed=key)
        else:
            action = jnp.argmax(pi.probs, axis=-1)
        return action[0], next_hstate

    def _run_pair(partner_params, fcp_params, partner_h0, fcp_h0, fcp_run_id, partner_run_id, seed, recipe_codes):
        episode_ids = jnp.arange(num_episodes, dtype=jnp.int32)
        reset_keys = jax.vmap(
            lambda ep_idx: _episode_key(seed, fcp_run_id, partner_run_id, ep_idx)
        )(episode_ids)
        obs, state = env_reset_vmapped(reset_keys)

        ctx_state = None
        if context_mode == "encoder" and context_manager is not None:
            recipe_ids = _recipe_labels(_recipe_bits_from_state(state), recipe_codes)
            ctx_state = context_manager.init_state(recipe_ids.astype(jnp.int32))

        total_reward = jnp.zeros((num_episodes,), dtype=jnp.float32)
        done = jnp.zeros((num_episodes,), dtype=jnp.bool_)
        pair_key = jax.random.PRNGKey(seed)
        pair_key = jax.random.fold_in(pair_key, fcp_run_id)
        pair_key = jax.random.fold_in(pair_key, partner_run_id)

        def _step(carry, _):
            obs, state, partner_h, fcp_h, ctx_state, total_reward, done, pair_key = carry

            pair_key, k_partner, k_fcp, k_env = jax.random.split(pair_key, 4)
            step_keys = jax.random.split(k_env, num_episodes)

            a0, partner_h = _policy_action(
                partner_policy,
                partner_params,
                obs["agent_0"],
                done,
                partner_h,
                k_partner,
            )

            context_vec = None
            if context_mode == "encoder" and ctx_state is not None:
                context_vec = ctx_state.recipe_ctx
            elif context_mode == "oracle":
                recipe_ids = _recipe_labels(_recipe_bits_from_state(state), recipe_codes)
                context_vec = jax.nn.one_hot(recipe_ids.astype(jnp.int32), len(recipe_codes))

            a1, fcp_h = _policy_action(
                fcp_policy,
                fcp_params,
                obs["agent_1"],
                done,
                fcp_h,
                k_fcp,
                context=context_vec,
            )

            actions = {
                "agent_0": jnp.asarray(a0, dtype=jnp.int32),
                "agent_1": jnp.asarray(a1, dtype=jnp.int32),
            }
            next_obs, next_state, reward, dones, _ = env_step_vmapped(step_keys, state, actions)

            active_mask = (~done).astype(reward["agent_1"].dtype)
            total_reward = total_reward + reward["agent_1"] * active_mask

            if context_mode == "encoder" and context_manager is not None and ctx_state is not None:
                recipe_id_next = _recipe_labels(_recipe_bits_from_state(next_state), recipe_codes)
                ctx_state = context_manager.update(
                    state=ctx_state,
                    ego_obs=jnp.asarray(obs["agent_1"]),
                    partner_act=jnp.asarray(a0, dtype=jnp.int32),
                    current_recipes=jnp.asarray(recipe_id_next, dtype=jnp.int32),
                    dones=jnp.asarray(dones["__all__"], dtype=jnp.bool_),
                )

            done = done | jnp.asarray(dones["__all__"], dtype=jnp.bool_)
            return (
                next_obs,
                next_state,
                partner_h,
                fcp_h,
                ctx_state,
                total_reward,
                done,
                pair_key,
            ), None

        carry0 = (obs, state, partner_h0, fcp_h0, ctx_state, total_reward, done, pair_key)
        carry, _ = jax.lax.scan(_step, carry0, None, length=episode_limit)
        return carry[5]

    return jax.jit(_run_pair)


def _run_pair_scanned(
    scanned_pair_runner: Callable,
    fcp_policy: ContextualPPOPolicy,
    partner_policy: ContextualPPOPolicy,
    fcp_run_id: int,
    partner_run_id: int,
    num_episodes: int,
    seed: int,
    recipe_codes,
):
    partner_h0 = partner_policy.init_hstate(batch_size=num_episodes)
    fcp_h0 = fcp_policy.init_hstate(batch_size=num_episodes)
    rewards = scanned_pair_runner(
        partner_policy.params,
        fcp_policy.params,
        partner_h0,
        fcp_h0,
        jnp.asarray(fcp_run_id, dtype=jnp.uint32),
        jnp.asarray(partner_run_id, dtype=jnp.uint32),
        jnp.asarray(seed, dtype=jnp.uint32),
        recipe_codes,
    )
    return np.asarray(rewards, dtype=np.float32)


def _run_pair_parallel(
    env_reset_vmapped,
    env_step_vmapped,
    fcp_policy: ContextualPPOPolicy,
    partner_policy: ContextualPPOPolicy,
    pair_key,
    fcp_run_id: int,
    partner_run_id: int,
    num_episodes: int,
    seed: int,
    recipe_codes,
    variant: VariantEvalConfig,
    context_manager: Optional[RecipeContextManager],
    env_max_steps: int,
    max_steps: Optional[int] = None,
):
    episode_ids = jnp.arange(num_episodes, dtype=jnp.int32)
    reset_keys = jax.vmap(
        lambda ep_idx: _episode_key(seed, fcp_run_id, partner_run_id, ep_idx)
    )(episode_ids)
    obs, state = env_reset_vmapped(reset_keys)

    fcp_h = fcp_policy.init_hstate(batch_size=num_episodes)
    partner_h = partner_policy.init_hstate(batch_size=num_episodes)

    ctx_state = None
    if variant.context_mode == "encoder" and context_manager is not None:
        recipe_ids = _recipe_labels(_recipe_bits_from_state(state), recipe_codes)
        ctx_state = context_manager.init_state(recipe_ids.astype(jnp.int32))

    total_reward = jnp.zeros((num_episodes,), dtype=jnp.float32)
    done = jnp.zeros((num_episodes,), dtype=jnp.bool_)
    episode_limit = int(max_steps) if max_steps is not None else int(env_max_steps)

    for _ in range(episode_limit):
        if bool(jnp.all(done)):
            break
        pair_key, k_partner, k_fcp, k_env = jax.random.split(pair_key, 4)
        step_keys = jax.random.split(k_env, num_episodes)

        # agent_0: partner
        a0, partner_h = partner_policy.compute_action(
            obs["agent_0"],
            done,
            partner_h,
            k_partner,
        )

        context_vec = None
        if variant.context_mode == "encoder" and ctx_state is not None:
            context_vec = ctx_state.recipe_ctx
        elif variant.context_mode == "oracle":
            recipe_ids = _recipe_labels(_recipe_bits_from_state(state), recipe_codes)
            context_vec = jax.nn.one_hot(recipe_ids.astype(jnp.int32), len(recipe_codes))

        # agent_1: evaluated FCP
        a1, fcp_h = fcp_policy.compute_action(
            obs["agent_1"],
            done,
            fcp_h,
            k_fcp,
            context=context_vec,
        )

        actions = {
            "agent_0": jnp.asarray(a0, dtype=jnp.int32),
            "agent_1": jnp.asarray(a1, dtype=jnp.int32),
        }

        next_obs, next_state, reward, dones, _ = env_step_vmapped(step_keys, state, actions)
        active_mask = (~done).astype(reward["agent_1"].dtype)
        total_reward = total_reward + reward["agent_1"] * active_mask

        if variant.context_mode == "encoder" and context_manager is not None and ctx_state is not None:
            recipe_id_next = _recipe_labels(_recipe_bits_from_state(next_state), recipe_codes)
            ctx_state = context_manager.update(
                state=ctx_state,
                ego_obs=jnp.asarray(obs["agent_1"]),
                partner_act=jnp.asarray(a0, dtype=jnp.int32),
                current_recipes=jnp.asarray(recipe_id_next, dtype=jnp.int32),
                dones=jnp.asarray(dones["__all__"], dtype=jnp.bool_),
            )

        obs, state = next_obs, next_state
        done = done | jnp.asarray(dones["__all__"], dtype=jnp.bool_)

    return np.asarray(total_reward, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp_dir", required=True, type=Path)
    parser.add_argument("--fcp_base_dir", type=Path, default=None)
    parser.add_argument("--fcp_encoder_dir", type=Path, default=None)
    parser.add_argument("--fcp_oracle_dir", type=Path, default=None)
    parser.add_argument("--encoder_ckpt", type=Path, default=None)
    parser.add_argument(
        "--variants",
        type=str,
        default="base,encoder,oracle",
        help="Comma-separated variants to evaluate. Choices: base,encoder,oracle",
    )
    parser.add_argument("--layout", default="demo_cook_simple")
    parser.add_argument("--agent_view_size", type=int, default=2)
    parser.add_argument("--episodes_per_pair", type=int, default=20)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional episode step cap for faster evaluation (defaults to env max_steps).",
    )
    parser.add_argument(
        "--encoder_k",
        type=int,
        default=None,
        help="Sequence length K for context manager. If omitted, infer from encoder/dataset metadata.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_csv", type=Path, default=Path("eval_results_fcp_variants.csv"))
    parser.add_argument("--summary_csv", type=Path, default=Path("eval_summary_fcp_variants.csv"))
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="Ignore previous CSVs and start from scratch.",
    )
    parser.add_argument(
        "--disable_scanned_rollout",
        action="store_true",
        help="Use the original Python timestep loop instead of the jitted lax.scan rollout.",
    )
    args = parser.parse_args()

    selected_variants = [v.strip().lower() for v in args.variants.split(",") if v.strip()]
    allowed_variants = {"base", "encoder", "oracle"}
    invalid_variants = [v for v in selected_variants if v not in allowed_variants]
    if invalid_variants:
        raise ValueError(f"Invalid variants: {invalid_variants}. Allowed: {sorted(allowed_variants)}")
    if not selected_variants:
        raise ValueError("No variants selected. Use --variants base,encoder,oracle (or subset).")

    if not args.sp_dir.exists():
        raise FileNotFoundError(f"Directory not found: {args.sp_dir}")
    if "base" in selected_variants:
        if args.fcp_base_dir is None or (not args.fcp_base_dir.exists()):
            raise FileNotFoundError(f"Base variant dir not found: {args.fcp_base_dir}")
    if "encoder" in selected_variants:
        if args.fcp_encoder_dir is None or (not args.fcp_encoder_dir.exists()):
            raise FileNotFoundError(f"Encoder variant dir not found: {args.fcp_encoder_dir}")
        if args.encoder_ckpt is None or (not args.encoder_ckpt.exists()):
            raise FileNotFoundError(f"Encoder checkpoint not found: {args.encoder_ckpt}")
    if "oracle" in selected_variants:
        if args.fcp_oracle_dir is None or (not args.fcp_oracle_dir.exists()):
            raise FileNotFoundError(f"Oracle variant dir not found: {args.fcp_oracle_dir}")

    env_kwargs = {
        "layout": args.layout,
        "agent_view_size": args.agent_view_size,
        "random_agent_positions": False,
        "sample_recipe_on_delivery": True,
        "negative_rewards": True,
    }
    env = jaxmarl.make("overcooked_v2", **env_kwargs)
    env = OvercookedV2LogWrapper(env, replace_info=False)
    env_reset_vmapped = jax.jit(jax.vmap(env.reset, in_axes=(0,)))
    env_step_vmapped = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))

    obs_shape = tuple(env.observation_space().shape)
    recipe_codes = jnp.array(_build_recipe_codebook(env), dtype=jnp.int32)
    num_classes = len(recipe_codes)

    context_manager = None
    encoder_use_actions = False
    encoder_action_dim = 0
    if "encoder" in selected_variants:
        encoder_meta_path = Path(str(args.encoder_ckpt) + ".meta.json")
        encoder_meta = _load_json_if_exists(encoder_meta_path)
        encoder_k = _resolve_encoder_k(encoder_meta_path, args.encoder_k)
        if encoder_meta:
            encoder_use_actions = bool(encoder_meta.get("use_actions", False))
            encoder_action_dim = int(encoder_meta.get("action_dim", 0))

        if "num_classes" in encoder_meta and int(encoder_meta["num_classes"]) != num_classes:
            print(
                "[Warn] encoder num_classes differs from env recipe count: "
                f"{encoder_meta['num_classes']} vs {num_classes}"
            )

        encoder_model, encoder_params = _load_encoder(
            encoder_ckpt=args.encoder_ckpt,
            obs_shape=obs_shape,
            num_classes=num_classes,
            k=encoder_k,
            use_actions=encoder_use_actions,
            action_dim=encoder_action_dim,
        )

        context_manager = RecipeContextManager(
            encoder_apply_fn=encoder_model.apply,
            encoder_params=encoder_params,
            K=encoder_k,
            num_envs=args.episodes_per_pair,
            num_actions=encoder_action_dim,
            num_classes=num_classes,
            obs_shape=obs_shape,
            mask_fn=get_mask_fn(num_ingredients=env.layout.num_ingredients),
            use_actions=encoder_use_actions,
        )

    variants = []
    if "base" in selected_variants:
        variants.append(VariantEvalConfig("base", args.fcp_base_dir, context_mode="none"))
    if "encoder" in selected_variants:
        variants.append(VariantEvalConfig("encoder", args.fcp_encoder_dir, context_mode="encoder"))
    if "oracle" in selected_variants:
        variants.append(VariantEvalConfig("oracle", args.fcp_oracle_dir, context_mode="oracle"))

    sp_run_ids = _list_run_ids(args.sp_dir)
    if not sp_run_ids:
        raise ValueError(f"No run_*/ckpt_final found in SP dir: {args.sp_dir}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite_output:
        if args.output_csv.exists():
            args.output_csv.unlink()
        if args.summary_csv.exists():
            args.summary_csv.unlink()

    existing_detail_df = _load_existing_detail(args.output_csv)
    existing_detail_df = existing_detail_df.dropna(subset=["variant", "fcp_run", "partner_run"])
    expected_max_steps = int(args.max_steps) if args.max_steps is not None else int(env.max_steps)
    completed_map = {}
    for _, r in existing_detail_df.iterrows():
        row_episodes = int(r["episodes"]) if not pd.isna(r["episodes"]) else int(args.episodes_per_pair)
        row_max_steps = int(r["max_steps"]) if not pd.isna(r["max_steps"]) else expected_max_steps
        row_seed = int(r["seed"]) if not pd.isna(r["seed"]) else int(args.seed)

        # Resume only when key settings match current run.
        if row_episodes != int(args.episodes_per_pair):
            continue
        if row_max_steps != expected_max_steps:
            continue
        if row_seed != int(args.seed):
            continue

        key = (str(r["variant"]), int(r["fcp_run"]), int(r["partner_run"]))
        completed_map[key] = {
            "variant": str(r["variant"]),
            "pair_order": int(r["pair_order"]) if not pd.isna(r["pair_order"]) else -1,
            "fcp_run": int(r["fcp_run"]),
            "partner_run": int(r["partner_run"]),
            "episodes": row_episodes,
            "max_steps": row_max_steps,
            "seed": row_seed,
            "mean_return": float(r["mean_return"]),
            "std_return": float(r["std_return"]),
        }

    rows = []
    summary_rows = []
    sp_policy_cache = {
        run_id: _load_policy(args.sp_dir, run_id, stochastic=args.stochastic)
        for run_id in sp_run_ids
    }

    for variant in variants:
        fcp_run_ids = _list_run_ids(variant.fcp_dir)
        if not fcp_run_ids:
            raise ValueError(f"No run_*/ckpt_final found in variant dir: {variant.fcp_dir}")

        print(f"\n[Eval] variant={variant.name} fcp_runs={len(fcp_run_ids)} partners={len(sp_run_ids)}")

        matrix = np.zeros((len(fcp_run_ids), len(sp_run_ids)), dtype=np.float32)
        fcp_policy_cache = {
            run_id: _load_policy(variant.fcp_dir, run_id, stochastic=args.stochastic)
            for run_id in fcp_run_ids
        }
        scanned_pair_runner = None
        if not args.disable_scanned_rollout:
            scanned_pair_runner = _make_scanned_pair_runner(
                env_reset_vmapped=env_reset_vmapped,
                env_step_vmapped=env_step_vmapped,
                fcp_policy=fcp_policy_cache[fcp_run_ids[0]],
                partner_policy=sp_policy_cache[sp_run_ids[0]],
                variant=variant,
                context_manager=context_manager if variant.context_mode == "encoder" else None,
                num_episodes=args.episodes_per_pair,
                episode_limit=expected_max_steps,
                stochastic=args.stochastic,
            )
            print(
                "[Fast rollout] using jitted lax.scan over "
                f"{expected_max_steps} timesteps; first new pair includes compile time."
            )
        ep_pbar = tqdm(
            total=len(fcp_run_ids) * len(sp_run_ids) * args.episodes_per_pair,
            desc=f"{variant.name} episodes",
            leave=True,
        )

        for i, fcp_run_id in enumerate(fcp_run_ids):
            fcp_policy = fcp_policy_cache[fcp_run_id]

            for j, sp_run_id in enumerate(sp_run_ids):
                pair_order = i * len(sp_run_ids) + j + 1
                resume_key = (variant.name, int(fcp_run_id), int(sp_run_id))

                if resume_key in completed_map:
                    cached = completed_map[resume_key]
                    matrix[i, j] = float(cached["mean_return"])
                    rows.append(cached)
                    ep_pbar.update(args.episodes_per_pair)
                    ep_pbar.set_postfix(
                        {
                            "status": "resume-skip",
                            "pair": pair_order,
                            "fcp": fcp_run_id,
                            "sp": sp_run_id,
                            "pair_mean": f"{float(cached['mean_return']):.1f}",
                        }
                    )
                    continue

                partner_policy = sp_policy_cache[sp_run_id]

                if scanned_pair_runner is None:
                    pair_key = jax.random.PRNGKey(args.seed)
                    pair_key = jax.random.fold_in(pair_key, int(fcp_run_id))
                    pair_key = jax.random.fold_in(pair_key, int(sp_run_id))

                    rewards = _run_pair_parallel(
                        env_reset_vmapped=env_reset_vmapped,
                        env_step_vmapped=env_step_vmapped,
                        fcp_policy=fcp_policy,
                        partner_policy=partner_policy,
                        pair_key=pair_key,
                        fcp_run_id=fcp_run_id,
                        partner_run_id=sp_run_id,
                        num_episodes=args.episodes_per_pair,
                        seed=args.seed,
                        recipe_codes=recipe_codes,
                        variant=variant,
                        context_manager=context_manager if variant.context_mode == "encoder" else None,
                        env_max_steps=env.max_steps,
                        max_steps=args.max_steps,
                    )
                else:
                    rewards = _run_pair_scanned(
                        scanned_pair_runner=scanned_pair_runner,
                        fcp_policy=fcp_policy,
                        partner_policy=partner_policy,
                        fcp_run_id=fcp_run_id,
                        partner_run_id=sp_run_id,
                        num_episodes=args.episodes_per_pair,
                        seed=args.seed,
                        recipe_codes=recipe_codes,
                    )
                ep_pbar.update(args.episodes_per_pair)
                ep_pbar.set_postfix(
                    {
                        "status": "saved",
                        "pair": pair_order,
                        "fcp": fcp_run_id,
                        "sp": sp_run_id,
                        "pair_mean": f"{float(np.mean(rewards)):.1f}",
                    }
                )

                mean_r = float(np.mean(rewards))
                std_r = float(np.std(rewards))
                matrix[i, j] = mean_r

                row = {
                    "variant": variant.name,
                    "pair_order": pair_order,
                    "fcp_run": int(fcp_run_id),
                    "partner_run": int(sp_run_id),
                    "episodes": int(args.episodes_per_pair),
                    "max_steps": int(args.max_steps) if args.max_steps is not None else int(env.max_steps),
                    "seed": int(args.seed),
                    "mean_return": mean_r,
                    "std_return": std_r,
                }
                rows.append(row)
                completed_map[resume_key] = row
                _append_detail_row(args.output_csv, row)
        ep_pbar.close()

        partner_means = np.mean(matrix, axis=0)
        global_mean = float(np.mean(matrix))
        global_std = float(np.std(matrix))

        for pj, sp_run_id in enumerate(sp_run_ids):
            summary_rows.append(
                {
                    "variant": variant.name,
                    "metric": "partner_mean",
                    "partner_run": sp_run_id,
                    "value": float(partner_means[pj]),
                }
            )

        summary_rows.append(
            {
                "variant": variant.name,
                "metric": "global_mean",
                "partner_run": -1,
                "value": global_mean,
            }
        )
        summary_rows.append(
            {
                "variant": variant.name,
                "metric": "global_std",
                "partner_run": -1,
                "value": global_std,
            }
        )

        print(f"[Result] {variant.name}: global_mean={global_mean:.3f}, global_std={global_std:.3f}")

    # Delta metrics
    summary_df = pd.DataFrame(summary_rows)
    if {"base", "encoder", "oracle"}.issubset(set(selected_variants)):
        get_global = lambda name: float(
            summary_df[
                (summary_df["variant"] == name) & (summary_df["metric"] == "global_mean")
            ]["value"].iloc[0]
        )

        base_mean = get_global("base")
        enc_mean = get_global("encoder")
        oracle_mean = get_global("oracle")

        delta_rows = [
            {"variant": "delta", "metric": "encoder_minus_base", "partner_run": -1, "value": enc_mean - base_mean},
            {"variant": "delta", "metric": "oracle_minus_base", "partner_run": -1, "value": oracle_mean - base_mean},
        ]

        headroom = oracle_mean - base_mean
        utilization = (enc_mean - base_mean) / headroom if headroom > 1e-8 else np.nan
        delta_rows.append(
            {"variant": "delta", "metric": "encoder_utilization_vs_oracle", "partner_run": -1, "value": utilization}
        )

        summary_df = pd.concat([summary_df, pd.DataFrame(delta_rows)], ignore_index=True)

    current_detail_df = pd.DataFrame(rows)
    merged_detail_df = pd.concat([existing_detail_df, current_detail_df], ignore_index=True)
    if not merged_detail_df.empty:
        merged_detail_df = merged_detail_df.sort_values(
            by=["variant", "pair_order", "fcp_run", "partner_run"]
        ).drop_duplicates(
            subset=["variant", "fcp_run", "partner_run"],
            keep="last",
        )

    detail_df = merged_detail_df
    detail_df.to_csv(args.output_csv, index=False)
    summary_df.to_csv(args.summary_csv, index=False)

    print(f"\nSaved detail csv:  {args.output_csv.resolve()}")
    print(f"Saved summary csv: {args.summary_csv.resolve()}")


if __name__ == "__main__":
    main()
