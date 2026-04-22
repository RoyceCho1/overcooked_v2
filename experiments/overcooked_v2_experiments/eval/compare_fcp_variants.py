import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd

import jaxmarl
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper
from jaxmarl.environments.overcooked_v2.common import DynamicObject

from overcooked_v2_experiments.ppo.policy import PPOPolicy
from overcooked_v2_experiments.ppo.utils.store import load_checkpoint
from overcooked_v2_experiments.recipe.context import RecipeContextManager
from overcooked_v2_experiments.recipe.masking import get_mask_fn
from overcooked_v2_experiments.recipe.recipe_encoder_jax import RecipeEncoder


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
    return ContextualPPOPolicy(params=params, config=config, stochastic=stochastic)


def _build_recipe_codebook(env) -> np.ndarray:
    recipes = jnp.array(env.layout.possible_recipes, dtype=jnp.int32)
    recipe_codes = jax.vmap(DynamicObject.get_recipe_encoding)(recipes)
    return np.array(recipe_codes, dtype=np.int32)


def _recipe_bits_from_state(state):
    # OvercookedV2LogWrapper keeps env state under `env_state`.
    if hasattr(state, "env_state"):
        return state.env_state.recipe
    return state.recipe


def _recipe_label(recipe_bits: int, recipe_codes: np.ndarray) -> int:
    matches = np.where(recipe_codes == int(recipe_bits))[0]
    if len(matches) == 0:
        return 0
    return int(matches[0])


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


def _episode_key(seed: int, fcp_run_id: int, partner_run_id: int, ep_idx: int):
    key = jax.random.PRNGKey(seed)
    key = jax.random.fold_in(key, int(fcp_run_id))
    key = jax.random.fold_in(key, int(partner_run_id))
    key = jax.random.fold_in(key, int(ep_idx))
    return key


def _run_episode(
    env,
    fcp_policy: ContextualPPOPolicy,
    partner_policy: ContextualPPOPolicy,
    key,
    recipe_codes: np.ndarray,
    variant: VariantEvalConfig,
    context_manager: Optional[RecipeContextManager],
):
    obs, state = env.reset(key)

    fcp_h = fcp_policy.init_hstate(batch_size=1)
    partner_h = partner_policy.init_hstate(batch_size=1)

    ctx_state = None
    if variant.context_mode == "encoder" and context_manager is not None:
        recipe_id = _recipe_label(_recipe_bits_from_state(state), recipe_codes)
        ctx_state = context_manager.init_state(jnp.array([recipe_id], dtype=jnp.int32))

    total_reward = 0.0
    done = False
    t = 0

    while not done and t < int(env.max_steps):
        key, k_partner, k_fcp, k_step = jax.random.split(key, 4)

        # agent_0: partner
        a0, partner_h = partner_policy.compute_action(
            obs["agent_0"],
            jnp.bool_(done),
            partner_h,
            k_partner,
        )

        context_vec = None
        if variant.context_mode == "encoder" and ctx_state is not None:
            context_vec = ctx_state.recipe_ctx[0]
        elif variant.context_mode == "oracle":
            recipe_id = _recipe_label(_recipe_bits_from_state(state), recipe_codes)
            context_vec = jax.nn.one_hot(jnp.array(recipe_id), len(recipe_codes))

        # agent_1: evaluated FCP
        a1, fcp_h = fcp_policy.compute_action(
            obs["agent_1"],
            jnp.bool_(done),
            fcp_h,
            k_fcp,
            context=context_vec,
        )

        actions = {
            "agent_0": int(a0),
            "agent_1": int(a1),
        }

        next_obs, next_state, reward, dones, _ = env.step(k_step, state, actions)
        total_reward += float(reward["agent_1"])

        if variant.context_mode == "encoder" and context_manager is not None and ctx_state is not None:
            recipe_id_next = _recipe_label(_recipe_bits_from_state(next_state), recipe_codes)
            ctx_state = context_manager.update(
                state=ctx_state,
                ego_obs=jnp.array(obs["agent_1"])[None, ...],
                partner_act=jnp.array([int(a0)]),
                current_recipes=jnp.array([recipe_id_next], dtype=jnp.int32),
                dones=jnp.array([bool(dones["__all__"])]),
            )

        obs, state = next_obs, next_state
        done = bool(dones["__all__"])
        t += 1

    return total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp_dir", required=True, type=Path)
    parser.add_argument("--fcp_base_dir", required=True, type=Path)
    parser.add_argument("--fcp_encoder_dir", required=True, type=Path)
    parser.add_argument("--fcp_oracle_dir", required=True, type=Path)
    parser.add_argument("--encoder_ckpt", required=True, type=Path)
    parser.add_argument("--layout", default="demo_cook_simple")
    parser.add_argument("--agent_view_size", type=int, default=2)
    parser.add_argument("--episodes_per_pair", type=int, default=20)
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
    args = parser.parse_args()

    for p in [args.sp_dir, args.fcp_base_dir, args.fcp_encoder_dir, args.fcp_oracle_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Directory not found: {p}")
    if not args.encoder_ckpt.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {args.encoder_ckpt}")

    env_kwargs = {
        "layout": args.layout,
        "agent_view_size": args.agent_view_size,
        "random_agent_positions": False,
        "sample_recipe_on_delivery": True,
        "negative_rewards": True,
    }
    env = jaxmarl.make("overcooked_v2", **env_kwargs)
    env = OvercookedV2LogWrapper(env, replace_info=False)

    obs_shape = tuple(env.observation_space().shape)
    recipe_codes = _build_recipe_codebook(env)
    num_classes = len(recipe_codes)

    encoder_meta_path = Path(str(args.encoder_ckpt) + ".meta.json")
    encoder_meta = _load_json_if_exists(encoder_meta_path)
    encoder_use_actions = False
    encoder_action_dim = 0
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
        num_envs=1,
        num_actions=encoder_action_dim,
        num_classes=num_classes,
        obs_shape=obs_shape,
        mask_fn=get_mask_fn(num_ingredients=env.layout.num_ingredients),
        use_actions=encoder_use_actions,
    )

    variants = [
        VariantEvalConfig("base", args.fcp_base_dir, context_mode="none"),
        VariantEvalConfig("encoder", args.fcp_encoder_dir, context_mode="encoder"),
        VariantEvalConfig("oracle", args.fcp_oracle_dir, context_mode="oracle"),
    ]

    sp_run_ids = _list_run_ids(args.sp_dir)
    if not sp_run_ids:
        raise ValueError(f"No run_*/ckpt_final found in SP dir: {args.sp_dir}")

    rows = []
    summary_rows = []

    for variant in variants:
        fcp_run_ids = _list_run_ids(variant.fcp_dir)
        if not fcp_run_ids:
            raise ValueError(f"No run_*/ckpt_final found in variant dir: {variant.fcp_dir}")

        print(f"\n[Eval] variant={variant.name} fcp_runs={len(fcp_run_ids)} partners={len(sp_run_ids)}")

        matrix = np.zeros((len(fcp_run_ids), len(sp_run_ids)), dtype=np.float32)

        for i, fcp_run_id in enumerate(fcp_run_ids):
            fcp_policy = _load_policy(variant.fcp_dir, fcp_run_id, stochastic=args.stochastic)

            for j, sp_run_id in enumerate(sp_run_ids):
                partner_policy = _load_policy(args.sp_dir, sp_run_id, stochastic=args.stochastic)

                rewards = []
                for ep_idx in range(args.episodes_per_pair):
                    ep_key = _episode_key(
                        seed=args.seed,
                        fcp_run_id=fcp_run_id,
                        partner_run_id=sp_run_id,
                        ep_idx=ep_idx,
                    )
                    r = _run_episode(
                        env=env,
                        fcp_policy=fcp_policy,
                        partner_policy=partner_policy,
                        key=ep_key,
                        recipe_codes=recipe_codes,
                        variant=variant,
                        context_manager=context_manager if variant.context_mode == "encoder" else None,
                    )
                    rewards.append(r)

                mean_r = float(np.mean(rewards))
                std_r = float(np.std(rewards))
                matrix[i, j] = mean_r

                rows.append(
                    {
                        "variant": variant.name,
                        "fcp_run": fcp_run_id,
                        "partner_run": sp_run_id,
                        "episodes": args.episodes_per_pair,
                        "mean_return": mean_r,
                        "std_return": std_r,
                    }
                )

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

    detail_df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(args.output_csv, index=False)
    summary_df.to_csv(args.summary_csv, index=False)

    print(f"\nSaved detail csv:  {args.output_csv.resolve()}")
    print(f"Saved summary csv: {args.summary_csv.resolve()}")


if __name__ == "__main__":
    main()
