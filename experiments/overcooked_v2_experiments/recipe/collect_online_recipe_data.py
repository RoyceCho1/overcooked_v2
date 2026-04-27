import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

import jaxmarl
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper

from overcooked_v2_experiments.eval.eval_online_recipe_accuracy import (
    _episode_key,
    _list_run_ids,
    _load_encoder,
    _load_json_if_exists,
    _load_policy,
    _recipe_bits_from_state,
    _recipe_labels,
    _resolve_encoder_k,
    _resolve_recipe_codebook,
)
from overcooked_v2_experiments.recipe.context import RecipeContextManager
from overcooked_v2_experiments.recipe.masking import get_mask_fn


PAIR_FILE_RE = re.compile(r"partner_(?P<partner>\d+)_fcp_(?P<fcp>\d+)\.npz$")
PAIR_SUMMARY_COLUMNS = [
    "fcp_run",
    "partner_run",
    "episodes",
    "max_steps",
    "segment_k",
    "segment_stride",
    "seed",
    "num_segments",
    "path",
]


@dataclass
class PairDataset:
    obs: np.ndarray
    recipe: np.ndarray
    partner: np.ndarray
    fcp_run: np.ndarray
    episode: np.ndarray
    segment_start: np.ndarray


def _pair_string(pair: Tuple[int, int]) -> str:
    return f"{int(pair[0])}:{int(pair[1])}"


def _parse_pair_string(value: str) -> Tuple[int, int]:
    fcp_run, partner_run = value.split(":", 1)
    return int(fcp_run), int(partner_run)


def _pair_file(save_dir: Path, fcp_run_id: int, partner_run_id: int) -> Path:
    return save_dir / f"partner_{int(partner_run_id)}_fcp_{int(fcp_run_id)}.npz"


def _discover_completed_pairs(save_dir: Path) -> Set[Tuple[int, int]]:
    completed = set()
    if not save_dir.exists():
        return completed

    for path in save_dir.glob("partner_*_fcp_*.npz"):
        match = PAIR_FILE_RE.match(path.name)
        if not match:
            continue
        completed.add((int(match.group("fcp")), int(match.group("partner"))))
    return completed


def _progress_settings(args, encoder_k: int, recipe_codes: np.ndarray, fcp_run_ids, sp_run_ids):
    return {
        "version": 1,
        "layout": str(args.layout),
        "agent_view_size": int(args.agent_view_size),
        "episodes_per_pair": int(args.episodes_per_pair),
        "max_steps": int(args.max_steps),
        "env_max_steps": int(args._resolved_env_max_steps),
        "segment_k": int(args.segment_k),
        "segment_stride": int(args.segment_stride),
        "seed": int(args.seed),
        "encoder_k": int(encoder_k),
        "stochastic": bool(args.stochastic),
        "sp_dir": str(args.sp_dir),
        "fcp_dir": str(args.fcp_dir),
        "encoder_ckpt": str(args.encoder_ckpt),
        "recipe_codes": [int(x) for x in recipe_codes.tolist()],
        "fcp_run_ids": [int(x) for x in fcp_run_ids],
        "sp_run_ids": [int(x) for x in sp_run_ids],
    }


def _load_progress(path: Path, expected_settings: dict) -> Optional[Set[Tuple[int, int]]]:
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        progress = json.load(f)

    settings = progress.get("settings", {})
    legacy_expected_settings = dict(expected_settings)
    legacy_expected_settings.pop("env_max_steps", None)
    if settings not in (expected_settings, legacy_expected_settings):
        raise ValueError(
            "Existing progress metadata does not match the current command. "
            f"Use --overwrite_output to start fresh, or write to a new save_dir. "
            f"Progress file: {path}"
        )

    return {
        _parse_pair_string(pair)
        for pair in progress.get("completed_pairs", [])
    }


def _write_progress(path: Path, settings: dict, completed_pairs: Set[Tuple[int, int]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": settings,
        "completed_pairs": [_pair_string(pair) for pair in sorted(completed_pairs)],
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def _write_npz_atomic(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def _write_dataset_meta(
    save_dir: Path,
    args,
    recipe_codes: np.ndarray,
    completed_pairs: Set[Tuple[int, int]],
    total_segments: int,
):
    meta = {
        "layout": args.layout,
        "agent_view_size": int(args.agent_view_size),
        "episodes_per_pair": int(args.episodes_per_pair),
        "max_steps": int(args.max_steps),
        "env_max_steps": int(args._resolved_env_max_steps),
        "segment_k": int(args.segment_k),
        "segment_stride": int(args.segment_stride),
        "num_partners": len({partner for _, partner in completed_pairs}),
        "num_fcp_runs": len({fcp for fcp, _ in completed_pairs}),
        "num_completed_pairs": len(completed_pairs),
        "num_total_segments": int(total_segments),
        "num_recipe_classes": int(len(recipe_codes)),
        "recipe_codes": [int(x) for x in recipe_codes.tolist()],
        "recipe_include_partner_action": False,
        "recipe_partner_action_visibility_aware": False,
        "recipe_action_dim": 0,
        "collection_policy": "fcp_encoder_rollout",
        "obs_source": "agent_1",
        "agent_0_role": "sp_partner",
        "agent_1_role": "fcp_encoder_policy",
        "sample_recipe_on_delivery": True,
    }
    meta_path = save_dir / "dataset_meta.json"
    tmp_path = meta_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(meta_path)


def _count_saved_segments(save_dir: Path) -> int:
    total = 0
    for path in save_dir.glob("partner_*_fcp_*.npz"):
        try:
            with np.load(path) as data:
                total += int(len(data["recipe"]))
        except Exception:
            continue
    return total


def _count_segments_in_file(path: Path) -> int:
    try:
        with np.load(path) as data:
            return int(len(data["recipe"]))
    except Exception:
        return 0


def _write_pair_summary_csv(save_dir: Path, args, completed_pairs: Set[Tuple[int, int]]):
    import csv

    path = save_dir / "collection_pairs.csv"
    rows = []
    for fcp_run_id, partner_run_id in sorted(completed_pairs):
        npz_path = _pair_file(save_dir, fcp_run_id, partner_run_id)
        if not npz_path.exists():
            continue
        rows.append(
            {
                "fcp_run": int(fcp_run_id),
                "partner_run": int(partner_run_id),
                "episodes": int(args.episodes_per_pair),
                "max_steps": int(args.max_steps),
                "segment_k": int(args.segment_k),
                "segment_stride": int(args.segment_stride),
                "seed": int(args.seed),
                "num_segments": _count_segments_in_file(npz_path),
                "path": str(npz_path),
            }
        )

    tmp_path = path.with_suffix(".csv.tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def _pair_dataset_from_arrays(
    obs_arr: np.ndarray,
    recipe_arr: np.ndarray,
    done_arr: np.ndarray,
    alive_arr: np.ndarray,
    context_manager: RecipeContextManager,
    fcp_run_id: int,
    partner_run_id: int,
    num_episodes: int,
    segment_k: int,
    segment_stride: int,
) -> PairDataset:
    if obs_arr.shape[0] == 0:
        empty_obs = np.zeros((0, segment_k, *context_manager.obs_shape), dtype=np.float32)
        empty_i = np.zeros((0,), dtype=np.int32)
        return PairDataset(empty_obs, empty_i, empty_i, empty_i, empty_i, empty_i)

    obs_arr = obs_arr.astype(np.float32)
    recipe_arr = recipe_arr.astype(np.int32)
    done_arr = done_arr.astype(bool)
    alive_arr = alive_arr.astype(bool)
    num_steps = obs_arr.shape[0]

    obs_segments = []
    recipe_segments = []
    partner_segments = []
    fcp_segments = []
    episode_segments = []
    segment_starts = []

    for ep in range(num_episodes):
        curr_start = 0
        while curr_start + segment_k <= num_steps:
            end = curr_start + segment_k
            seg_done = done_arr[curr_start:end, ep]
            seg_alive = alive_arr[curr_start:end, ep]
            seg_recipe = recipe_arr[curr_start:end, ep]

            if np.any(seg_done) or not np.all(seg_alive):
                curr_start += segment_stride
                continue

            if not np.all(seg_recipe == seg_recipe[0]):
                curr_start += 1
                continue

            obs_segments.append(obs_arr[curr_start:end, ep])
            recipe_segments.append(int(seg_recipe[0]))
            partner_segments.append(int(partner_run_id))
            fcp_segments.append(int(fcp_run_id))
            episode_segments.append(int(ep))
            segment_starts.append(int(curr_start))
            curr_start += segment_stride

    if not obs_segments:
        empty_obs = np.zeros((0, segment_k, *context_manager.obs_shape), dtype=np.float32)
        empty_i = np.zeros((0,), dtype=np.int32)
        return PairDataset(empty_obs, empty_i, empty_i, empty_i, empty_i, empty_i)

    return PairDataset(
        obs=np.asarray(obs_segments, dtype=np.float32),
        recipe=np.asarray(recipe_segments, dtype=np.int32),
        partner=np.asarray(partner_segments, dtype=np.int32),
        fcp_run=np.asarray(fcp_segments, dtype=np.int32),
        episode=np.asarray(episode_segments, dtype=np.int32),
        segment_start=np.asarray(segment_starts, dtype=np.int32),
    )


def _make_scanned_pair_collector(
    env_reset_vmapped,
    env_step_vmapped,
    fcp_policy,
    partner_policy,
    context_manager: RecipeContextManager,
    mask_fn,
    num_episodes: int,
    episode_limit: int,
    stochastic: bool,
) -> Callable:
    """Build a jitted online-data rollout that scans timesteps on-device."""

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

    def _collect(partner_params, fcp_params, partner_h0, fcp_h0, fcp_run_id, partner_run_id, seed, recipe_codes):
        episode_ids = jnp.arange(num_episodes, dtype=jnp.int32)
        reset_keys = jax.vmap(
            lambda ep_idx: _episode_key(seed, fcp_run_id, partner_run_id, ep_idx)
        )(episode_ids)
        obs, state = env_reset_vmapped(reset_keys)

        initial_recipe = _recipe_labels(_recipe_bits_from_state(state), recipe_codes)
        ctx_state = context_manager.init_state(initial_recipe.astype(jnp.int32))

        done = jnp.zeros((num_episodes,), dtype=jnp.bool_)
        pair_key = jax.random.PRNGKey(seed)
        pair_key = jax.random.fold_in(pair_key, fcp_run_id)
        pair_key = jax.random.fold_in(pair_key, partner_run_id)

        def _step(carry, _):
            obs, state, partner_h, fcp_h, ctx_state, done, pair_key = carry

            alive = ~done
            true_recipe_code = _recipe_bits_from_state(state).astype(jnp.int32)
            true_recipe = _recipe_labels(true_recipe_code, recipe_codes).astype(jnp.int32)
            ego_obs = mask_fn(jnp.asarray(obs["agent_1"]))

            pair_key, k_partner, k_fcp, k_env = jax.random.split(pair_key, 4)
            a0, partner_h = _policy_action(
                partner_policy,
                partner_params,
                obs["agent_0"],
                done,
                partner_h,
                k_partner,
            )
            a1, fcp_h = _policy_action(
                fcp_policy,
                fcp_params,
                obs["agent_1"],
                done,
                fcp_h,
                k_fcp,
                context=ctx_state.recipe_ctx,
            )

            step_keys = jax.random.split(k_env, num_episodes)
            actions = {
                "agent_0": jnp.asarray(a0, dtype=jnp.int32),
                "agent_1": jnp.asarray(a1, dtype=jnp.int32),
            }
            next_obs, next_state, _, dones, _ = env_step_vmapped(step_keys, state, actions)

            next_recipe = _recipe_labels(_recipe_bits_from_state(next_state), recipe_codes)
            ctx_state = context_manager.update(
                state=ctx_state,
                ego_obs=jnp.asarray(obs["agent_1"]),
                partner_act=jnp.asarray(a0, dtype=jnp.int32),
                current_recipes=jnp.asarray(true_recipe, dtype=jnp.int32),
                dones=jnp.asarray(dones["__all__"], dtype=jnp.bool_),
                next_recipes=jnp.asarray(next_recipe, dtype=jnp.int32),
            )
            next_done = done | jnp.asarray(dones["__all__"], dtype=jnp.bool_)

            next_carry = (next_obs, next_state, partner_h, fcp_h, ctx_state, next_done, pair_key)
            records = {
                "obs": ego_obs,
                "recipe": true_recipe,
                "done": dones["__all__"],
                "alive": alive,
            }
            return next_carry, records

        carry0 = (obs, state, partner_h0, fcp_h0, ctx_state, done, pair_key)
        _, records = jax.lax.scan(_step, carry0, None, length=episode_limit)
        return records

    return jax.jit(_collect)


def _collect_pair_dataset_scanned(
    scanned_collector: Callable,
    fcp_policy,
    partner_policy,
    context_manager: RecipeContextManager,
    fcp_run_id: int,
    partner_run_id: int,
    num_episodes: int,
    seed: int,
    recipe_codes,
    segment_k: int,
    segment_stride: int,
) -> PairDataset:
    partner_h0 = partner_policy.init_hstate(batch_size=num_episodes)
    fcp_h0 = fcp_policy.init_hstate(batch_size=num_episodes)
    records = scanned_collector(
        partner_policy.params,
        fcp_policy.params,
        partner_h0,
        fcp_h0,
        jnp.asarray(fcp_run_id, dtype=jnp.uint32),
        jnp.asarray(partner_run_id, dtype=jnp.uint32),
        jnp.asarray(seed, dtype=jnp.uint32),
        recipe_codes,
    )
    records = jax.tree_util.tree_map(np.asarray, records)

    any_alive = np.any(records["alive"].astype(bool), axis=1)
    stopped = np.where(~any_alive)[0]
    if len(stopped) > 0:
        num_steps = int(stopped[0])
        records = {name: values[:num_steps] for name, values in records.items()}

    return _pair_dataset_from_arrays(
        obs_arr=records["obs"],
        recipe_arr=records["recipe"],
        done_arr=records["done"],
        alive_arr=records["alive"],
        context_manager=context_manager,
        fcp_run_id=fcp_run_id,
        partner_run_id=partner_run_id,
        num_episodes=num_episodes,
        segment_k=segment_k,
        segment_stride=segment_stride,
    )


def _collect_pair_dataset(
    env_reset_vmapped,
    env_step_vmapped,
    fcp_policy,
    partner_policy,
    context_manager: RecipeContextManager,
    mask_fn,
    fcp_run_id: int,
    partner_run_id: int,
    num_episodes: int,
    seed: int,
    recipe_codes,
    env_max_steps: int,
    max_steps: int,
    segment_k: int,
    segment_stride: int,
) -> PairDataset:
    episode_ids = jnp.arange(num_episodes, dtype=jnp.int32)
    reset_keys = jax.vmap(
        lambda ep_idx: _episode_key(seed, fcp_run_id, partner_run_id, ep_idx)
    )(episode_ids)
    obs, state = env_reset_vmapped(reset_keys)

    fcp_h = fcp_policy.init_hstate(batch_size=num_episodes)
    partner_h = partner_policy.init_hstate(batch_size=num_episodes)

    initial_recipe = _recipe_labels(_recipe_bits_from_state(state), recipe_codes)
    ctx_state = context_manager.init_state(initial_recipe.astype(jnp.int32))

    episode_limit = int(max_steps) if max_steps is not None else int(env_max_steps)
    done = jnp.zeros((num_episodes,), dtype=jnp.bool_)
    pair_key = jax.random.PRNGKey(seed)
    pair_key = jax.random.fold_in(pair_key, int(fcp_run_id))
    pair_key = jax.random.fold_in(pair_key, int(partner_run_id))

    obs_records = []
    recipe_records = []
    recipe_code_records = []
    done_records = []
    alive_records = []

    for _ in range(episode_limit):
        if bool(jnp.all(done)):
            break

        alive = ~done
        true_recipe_code = _recipe_bits_from_state(state).astype(jnp.int32)
        true_recipe = _recipe_labels(true_recipe_code, recipe_codes).astype(jnp.int32)
        ego_obs = mask_fn(jnp.asarray(obs["agent_1"]))

        pair_key, k_partner, k_fcp, k_env = jax.random.split(pair_key, 4)
        a0, partner_h = partner_policy.compute_action(
            obs["agent_0"],
            done,
            partner_h,
            k_partner,
        )
        a1, fcp_h = fcp_policy.compute_action(
            obs["agent_1"],
            done,
            fcp_h,
            k_fcp,
            context=ctx_state.recipe_ctx,
        )

        step_keys = jax.random.split(k_env, num_episodes)
        actions = {
            "agent_0": jnp.asarray(a0, dtype=jnp.int32),
            "agent_1": jnp.asarray(a1, dtype=jnp.int32),
        }
        next_obs, next_state, _, dones, _ = env_step_vmapped(step_keys, state, actions)

        obs_records.append(np.asarray(ego_obs))
        recipe_records.append(np.asarray(true_recipe))
        recipe_code_records.append(np.asarray(true_recipe_code))
        done_records.append(np.asarray(dones["__all__"]))
        alive_records.append(np.asarray(alive))

        next_recipe = _recipe_labels(_recipe_bits_from_state(next_state), recipe_codes)
        ctx_state = context_manager.update(
            state=ctx_state,
            ego_obs=jnp.asarray(obs["agent_1"]),
            partner_act=jnp.asarray(a0, dtype=jnp.int32),
            current_recipes=jnp.asarray(true_recipe, dtype=jnp.int32),
            dones=jnp.asarray(dones["__all__"], dtype=jnp.bool_),
            next_recipes=jnp.asarray(next_recipe, dtype=jnp.int32),
        )

        obs, state = next_obs, next_state
        done = done | jnp.asarray(dones["__all__"], dtype=jnp.bool_)

    obs_arr = np.stack(obs_records, axis=0) if obs_records else np.zeros(
        (0, *context_manager.obs_shape),
        dtype=np.float32,
    )
    recipe_arr = np.stack(recipe_records, axis=0) if recipe_records else np.zeros(
        (0, num_episodes),
        dtype=np.int32,
    )
    done_arr = np.stack(done_records, axis=0) if done_records else np.zeros(
        (0, num_episodes),
        dtype=bool,
    )
    alive_arr = np.stack(alive_records, axis=0) if alive_records else np.zeros(
        (0, num_episodes),
        dtype=bool,
    )

    return _pair_dataset_from_arrays(
        obs_arr=obs_arr,
        recipe_arr=recipe_arr,
        done_arr=done_arr,
        alive_arr=alive_arr,
        context_manager=context_manager,
        fcp_run_id=fcp_run_id,
        partner_run_id=partner_run_id,
        num_episodes=num_episodes,
        segment_k=segment_k,
        segment_stride=segment_stride,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp_dir", required=True, type=Path)
    parser.add_argument("--fcp_dir", required=True, type=Path)
    parser.add_argument("--encoder_ckpt", required=True, type=Path)
    parser.add_argument(
        "--save_dir",
        default=Path("./runs/k_sweep_demo_cook_simple/k6/recipe_data_k6_online_fcp_obs_only"),
        type=Path,
    )
    parser.add_argument("--layout", default="demo_cook_simple")
    parser.add_argument("--agent_view_size", type=int, default=2)
    parser.add_argument("--episodes_per_pair", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument(
        "--env_max_steps",
        type=int,
        default=None,
        help="Episode horizon passed to OvercookedV2. Defaults to max(--max_steps, 400).",
    )
    parser.add_argument("--segment_k", type=int, default=6)
    parser.add_argument("--segment_stride", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress_json", default=None, type=Path)
    parser.add_argument("--overwrite_output", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--disable_scanned_rollout",
        action="store_true",
        help="Use the original Python timestep loop instead of the jitted lax.scan rollout.",
    )
    args = parser.parse_args()
    args._resolved_env_max_steps = (
        int(args.env_max_steps)
        if args.env_max_steps is not None
        else max(int(args.max_steps), 400)
    )
    if args._resolved_env_max_steps < int(args.max_steps):
        raise ValueError(
            f"--env_max_steps ({args._resolved_env_max_steps}) must be >= --max_steps ({args.max_steps})."
        )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.progress_json or (args.save_dir / "collection_progress.json")

    if args.overwrite_output:
        for path in args.save_dir.glob("partner_*_fcp_*.npz"):
            path.unlink()
        for path in [
            progress_path,
            args.save_dir / "dataset_meta.json",
            args.save_dir / "collection_pairs.csv",
        ]:
            if path.exists():
                path.unlink()

    env_kwargs = {
        "layout": args.layout,
        "agent_view_size": args.agent_view_size,
        "random_agent_positions": False,
        "sample_recipe_on_delivery": True,
        "negative_rewards": True,
        "max_steps": args._resolved_env_max_steps,
    }
    env = OvercookedV2LogWrapper(
        jaxmarl.make("overcooked_v2", **env_kwargs),
        replace_info=False,
    )
    env_reset_vmapped = jax.jit(jax.vmap(env.reset, in_axes=(0,)))
    env_step_vmapped = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))

    obs_shape = tuple(env.observation_space().shape)
    encoder_meta_path = Path(str(args.encoder_ckpt) + ".meta.json")
    encoder_meta = _load_json_if_exists(encoder_meta_path)
    recipe_codes_np = _resolve_recipe_codebook(env, encoder_meta)
    recipe_codes = jnp.asarray(recipe_codes_np, dtype=jnp.int32)
    num_classes = int(len(recipe_codes_np))
    encoder_k = _resolve_encoder_k(encoder_meta_path, None)
    encoder_use_actions = bool(encoder_meta.get("use_actions", False))
    encoder_action_dim = int(encoder_meta.get("action_dim", 0))

    if encoder_use_actions:
        raise ValueError(
            "collect_online_recipe_data.py currently writes obs-only datasets, "
            "but the supplied encoder checkpoint expects actions."
        )

    encoder_model, encoder_params = _load_encoder(
        encoder_ckpt=args.encoder_ckpt,
        obs_shape=obs_shape,
        num_classes=num_classes,
        k=encoder_k,
        use_actions=encoder_use_actions,
        action_dim=encoder_action_dim,
    )

    mask_fn = get_mask_fn(num_ingredients=env.layout.num_ingredients)
    context_manager = RecipeContextManager(
        encoder_apply_fn=encoder_model.apply,
        encoder_params=encoder_params,
        K=encoder_k,
        num_envs=args.episodes_per_pair,
        num_actions=encoder_action_dim,
        num_classes=num_classes,
        obs_shape=obs_shape,
        mask_fn=mask_fn,
        use_actions=encoder_use_actions,
    )

    sp_run_ids = _list_run_ids(args.sp_dir)
    fcp_run_ids = _list_run_ids(args.fcp_dir)
    if not sp_run_ids:
        raise ValueError(f"No SP run_*/ckpt_final found in {args.sp_dir}")
    if not fcp_run_ids:
        raise ValueError(f"No FCP run_*/ckpt_final found in {args.fcp_dir}")

    expected_pairs = {
        (int(fcp_run_id), int(sp_run_id))
        for fcp_run_id in fcp_run_ids
        for sp_run_id in sp_run_ids
    }
    progress_settings = _progress_settings(
        args=args,
        encoder_k=encoder_k,
        recipe_codes=recipe_codes_np,
        fcp_run_ids=fcp_run_ids,
        sp_run_ids=sp_run_ids,
    )
    progress_completed = _load_progress(progress_path, progress_settings)
    file_completed = _discover_completed_pairs(args.save_dir)
    completed_pairs = (
        file_completed if progress_completed is None else progress_completed | file_completed
    ) & expected_pairs

    _write_progress(progress_path, progress_settings, completed_pairs)
    _write_dataset_meta(
        args.save_dir,
        args,
        recipe_codes_np,
        completed_pairs,
        total_segments=_count_saved_segments(args.save_dir),
    )
    _write_pair_summary_csv(args.save_dir, args, completed_pairs)

    print("--- Online Recipe Dataset Collection ---")
    print(f"save_dir: {args.save_dir}")
    print(f"recipe_codes: {recipe_codes_np.tolist()}")
    print(f"encoder_k: {encoder_k}, segment_k: {args.segment_k}")
    print(f"completed pairs: {len(completed_pairs)}/{len(expected_pairs)}")
    print("agent_0=SP partner, agent_1=FCP encoder policy, obs_source=agent_1")
    print("----------------------------------------")

    sp_policy_cache = {
        run_id: _load_policy(args.sp_dir, run_id, stochastic=args.stochastic)
        for run_id in sp_run_ids
    }
    fcp_policy_cache = {
        run_id: _load_policy(args.fcp_dir, run_id, stochastic=args.stochastic)
        for run_id in fcp_run_ids
    }

    scanned_collector = None
    if not args.disable_scanned_rollout:
        scanned_collector = _make_scanned_pair_collector(
            env_reset_vmapped=env_reset_vmapped,
            env_step_vmapped=env_step_vmapped,
            fcp_policy=fcp_policy_cache[fcp_run_ids[0]],
            partner_policy=sp_policy_cache[sp_run_ids[0]],
            context_manager=context_manager,
            mask_fn=mask_fn,
            num_episodes=args.episodes_per_pair,
            episode_limit=int(args.max_steps),
            stochastic=args.stochastic,
        )
        print(
            "[Fast rollout] using jitted lax.scan over "
            f"{int(args.max_steps)} timesteps; first new pair includes compile time."
        )

    pbar = tqdm(total=len(expected_pairs), desc="online dataset pairs")
    pbar.update(len(completed_pairs))
    for fcp_idx, fcp_run_id in enumerate(fcp_run_ids):
        for sp_idx, sp_run_id in enumerate(sp_run_ids):
            pair = (int(fcp_run_id), int(sp_run_id))
            pair_order = fcp_idx * len(sp_run_ids) + sp_idx + 1
            if pair in completed_pairs:
                pbar.set_postfix(
                    {
                        "status": "resume-skip",
                        "pair": pair_order,
                        "fcp": fcp_run_id,
                        "sp": sp_run_id,
                    }
                )
                continue

            if scanned_collector is None:
                pair_data = _collect_pair_dataset(
                    env_reset_vmapped=env_reset_vmapped,
                    env_step_vmapped=env_step_vmapped,
                    fcp_policy=fcp_policy_cache[fcp_run_id],
                    partner_policy=sp_policy_cache[sp_run_id],
                    context_manager=context_manager,
                    mask_fn=mask_fn,
                    fcp_run_id=fcp_run_id,
                    partner_run_id=sp_run_id,
                    num_episodes=args.episodes_per_pair,
                    seed=args.seed,
                    recipe_codes=recipe_codes,
                    env_max_steps=int(env.max_steps),
                    max_steps=int(args.max_steps),
                    segment_k=int(args.segment_k),
                    segment_stride=int(args.segment_stride),
                )
            else:
                pair_data = _collect_pair_dataset_scanned(
                    scanned_collector=scanned_collector,
                    fcp_policy=fcp_policy_cache[fcp_run_id],
                    partner_policy=sp_policy_cache[sp_run_id],
                    context_manager=context_manager,
                    fcp_run_id=fcp_run_id,
                    partner_run_id=sp_run_id,
                    num_episodes=args.episodes_per_pair,
                    seed=args.seed,
                    recipe_codes=recipe_codes,
                    segment_k=int(args.segment_k),
                    segment_stride=int(args.segment_stride),
                )

            out_file = _pair_file(args.save_dir, fcp_run_id, sp_run_id)
            _write_npz_atomic(
                out_file,
                obs=pair_data.obs,
                recipe=pair_data.recipe,
                partner=pair_data.partner,
                fcp_run=pair_data.fcp_run,
                episode=pair_data.episode,
                segment_start=pair_data.segment_start,
            )

            completed_pairs.add(pair)
            total_segments = _count_saved_segments(args.save_dir)
            _write_progress(progress_path, progress_settings, completed_pairs)
            _write_dataset_meta(
                args.save_dir,
                args,
                recipe_codes_np,
                completed_pairs,
                total_segments=total_segments,
            )
            _write_pair_summary_csv(args.save_dir, args, completed_pairs)

            pbar.update(1)
            pbar.set_postfix(
                {
                    "status": "saved",
                    "pair": pair_order,
                    "fcp": fcp_run_id,
                    "sp": sp_run_id,
                    "segments": len(pair_data.recipe),
                }
            )
    pbar.close()

    total_segments = _count_saved_segments(args.save_dir)
    _write_dataset_meta(
        args.save_dir,
        args,
        recipe_codes_np,
        completed_pairs,
        total_segments=total_segments,
    )
    _write_pair_summary_csv(args.save_dir, args, completed_pairs)

    print(f"Saved online dataset to: {args.save_dir.resolve()}")
    print(f"Completed pairs: {len(completed_pairs)}/{len(expected_pairs)}")
    print(f"Total segments: {total_segments}")
    print(f"Progress JSON: {progress_path.resolve()}")


if __name__ == "__main__":
    main()
