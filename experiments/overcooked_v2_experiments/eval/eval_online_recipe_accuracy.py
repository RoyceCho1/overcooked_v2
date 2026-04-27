import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from tqdm import tqdm

import jaxmarl
from jaxmarl.environments.overcooked_v2.common import DynamicObject
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper

from overcooked_v2_experiments.ppo.policy import PPOPolicy
from overcooked_v2_experiments.ppo.utils.store import load_checkpoint
from overcooked_v2_experiments.recipe.context import RecipeContextManager
from overcooked_v2_experiments.recipe.masking import get_mask_fn
from overcooked_v2_experiments.recipe.recipe_encoder_jax import RecipeEncoder


TIMESTEP_COLUMNS = [
    "k",
    "fcp_run",
    "partner_run",
    "episode",
    "timestep",
    "valid",
    "true_recipe",
    "true_recipe_code",
    "pred_recipe",
    "pred_recipe_code",
    "correct",
    "swapped_correct",
    "recipe_changed",
    "max_prob",
    "prob_true_recipe",
    "episode_alive",
    "reward_t",
    "return_so_far",
]

EPISODE_COLUMNS = [
    "k",
    "fcp_run",
    "partner_run",
    "episode",
    "episode_return",
    "first_valid_timestep",
    "first_correct_timestep",
    "mean_valid_accuracy",
    "early_valid_accuracy",
    "late_valid_accuracy",
    "effective_correct_rate",
    "mean_prob_true_recipe",
    "flip_rate",
]

SUMMARY_COLUMNS = [
    "k",
    "metric",
    "group",
    "fcp_run",
    "partner_run",
    "recipe",
    "timestep",
    "value",
    "n",
]


@dataclass
class PairRollout:
    timestep_rows: List[dict]
    episode_rows: List[dict]


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


def _load_json_if_exists(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_recipe_codebook(env) -> np.ndarray:
    recipes = jnp.array(env.layout.possible_recipes, dtype=jnp.int32)
    recipe_codes = jax.vmap(DynamicObject.get_recipe_encoding)(recipes)
    return np.array(recipe_codes, dtype=np.int32)


def _resolve_recipe_codebook(env, encoder_meta: dict) -> np.ndarray:
    """Prefer the encoder dataset codebook so class ids match training labels."""
    data_dir = encoder_meta.get("data_dir", None)
    if data_dir:
        dataset_meta = _load_json_if_exists(Path(data_dir) / "dataset_meta.json")
        recipe_codes = dataset_meta.get("recipe_codes", None)
        if recipe_codes:
            return np.array(recipe_codes, dtype=np.int32)

    recipe_codes = encoder_meta.get("recipe_codes", None)
    if recipe_codes:
        return np.array(recipe_codes, dtype=np.int32)

    return _build_recipe_codebook(env)


def _recipe_bits_from_state(state):
    if hasattr(state, "env_state"):
        return state.env_state.recipe
    return state.recipe


def _recipe_label(recipe_bits, recipe_codes):
    matches = recipe_codes == recipe_bits
    has_match = jnp.any(matches)
    label = jnp.argmax(matches.astype(jnp.int32)).astype(jnp.int32)
    return jnp.where(has_match, label, jnp.int32(0))


def _recipe_labels(recipe_bits, recipe_codes):
    recipe_bits = jnp.asarray(recipe_bits)
    if recipe_bits.ndim == 0:
        return _recipe_label(recipe_bits, recipe_codes)
    return jax.vmap(lambda bits: _recipe_label(bits, recipe_codes))(recipe_bits)


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
        dataset_meta = _load_json_if_exists(Path(data_dir) / "dataset_meta.json")
        if "segment_k" in dataset_meta:
            return int(dataset_meta["segment_k"])

    return 10


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

    params = ocp.PyTreeCheckpointer().restore(str(encoder_ckpt), item=dummy_params)
    return model, params


def _episode_key(seed: int, fcp_run_id: int, partner_run_id: int, ep_idx: int):
    key = jax.random.PRNGKey(seed)
    key = jax.random.fold_in(key, jnp.asarray(fcp_run_id, dtype=jnp.uint32))
    key = jax.random.fold_in(key, jnp.asarray(partner_run_id, dtype=jnp.uint32))
    key = jax.random.fold_in(key, jnp.asarray(ep_idx, dtype=jnp.uint32))
    return key


def _nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def _first_index(mask: np.ndarray) -> int:
    idx = np.where(mask)[0]
    return int(idx[0]) if len(idx) > 0 else -1


def _flip_rate(pred: np.ndarray, valid: np.ndarray) -> float:
    pred_valid = pred[valid]
    if len(pred_valid) <= 1:
        return float("nan")
    return float(np.sum(pred_valid[1:] != pred_valid[:-1]) / len(pred_valid))


def _pair_rollout_from_arrays(
    arrays: Dict[str, np.ndarray],
    k: int,
    fcp_run_id: int,
    partner_run_id: int,
    num_episodes: int,
) -> PairRollout:
    num_steps = arrays["valid"].shape[0]
    timestep_rows = []
    episode_rows = []

    for ep in range(num_episodes):
        alive = arrays["episode_alive"][:, ep].astype(bool)
        valid = arrays["valid"][:, ep].astype(bool) & alive
        correct = arrays["correct"][:, ep].astype(bool) & alive
        pred = arrays["pred_recipe"][:, ep].astype(np.int32)
        prob_true = arrays["prob_true_recipe"][:, ep].astype(np.float32)
        episode_return = float(arrays["return_so_far"][-1, ep]) if num_steps else 0.0

        first_valid = _first_index(valid)
        first_correct = _first_index(correct)
        valid_accuracy = _nanmean(correct[valid].astype(np.float32))
        effective_correct = float(np.sum(correct) / max(np.sum(alive), 1))

        if first_valid >= 0:
            early_mask = valid & (np.arange(num_steps) >= first_valid) & (np.arange(num_steps) < first_valid + 20)
        else:
            early_mask = np.zeros((num_steps,), dtype=bool)
        late_mask = valid & (np.arange(num_steps) >= 100)

        episode_rows.append(
            {
                "k": k,
                "fcp_run": fcp_run_id,
                "partner_run": partner_run_id,
                "episode": ep,
                "episode_return": episode_return,
                "first_valid_timestep": first_valid,
                "first_correct_timestep": first_correct,
                "mean_valid_accuracy": valid_accuracy,
                "early_valid_accuracy": _nanmean(correct[early_mask].astype(np.float32)),
                "late_valid_accuracy": _nanmean(correct[late_mask].astype(np.float32)),
                "effective_correct_rate": effective_correct,
                "mean_prob_true_recipe": _nanmean(prob_true[valid]),
                "flip_rate": _flip_rate(pred, valid),
            }
        )

        for t in range(num_steps):
            timestep_rows.append(
                {
                    "k": k,
                    "fcp_run": fcp_run_id,
                    "partner_run": partner_run_id,
                    "episode": ep,
                    "timestep": t,
                    "valid": int(bool(arrays["valid"][t, ep] and arrays["episode_alive"][t, ep])),
                    "true_recipe": int(arrays["true_recipe"][t, ep]),
                    "true_recipe_code": int(arrays["true_recipe_code"][t, ep]),
                    "pred_recipe": int(arrays["pred_recipe"][t, ep]),
                    "pred_recipe_code": int(arrays["pred_recipe_code"][t, ep]),
                    "correct": int(bool(arrays["correct"][t, ep] and arrays["episode_alive"][t, ep])),
                    "swapped_correct": int(bool(arrays["swapped_correct"][t, ep] and arrays["episode_alive"][t, ep])),
                    "recipe_changed": int(bool(arrays["recipe_changed"][t, ep])),
                    "max_prob": float(arrays["max_prob"][t, ep]),
                    "prob_true_recipe": float(arrays["prob_true_recipe"][t, ep]),
                    "episode_alive": int(bool(arrays["episode_alive"][t, ep])),
                    "reward_t": float(arrays["reward_t"][t, ep]),
                    "return_so_far": float(arrays["return_so_far"][t, ep]),
                }
            )

    return PairRollout(timestep_rows=timestep_rows, episode_rows=episode_rows)


def _make_scanned_online_accuracy_rollout(
    env_reset_vmapped,
    env_step_vmapped,
    fcp_policy: ContextualPPOPolicy,
    partner_policy: ContextualPPOPolicy,
    context_manager: RecipeContextManager,
    num_episodes: int,
    episode_limit: int,
    stochastic: bool,
) -> Callable:
    """Build a jitted pair rollout that scans timesteps on-device."""

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

    def _rollout(partner_params, fcp_params, partner_h0, fcp_h0, fcp_run_id, partner_run_id, seed, recipe_codes):
        episode_ids = jnp.arange(num_episodes, dtype=jnp.int32)
        reset_keys = jax.vmap(
            lambda ep_idx: _episode_key(seed, fcp_run_id, partner_run_id, ep_idx)
        )(episode_ids)
        obs, state = env_reset_vmapped(reset_keys)

        initial_recipe = _recipe_labels(_recipe_bits_from_state(state), recipe_codes)
        ctx_state = context_manager.init_state(initial_recipe.astype(jnp.int32))

        done = jnp.zeros((num_episodes,), dtype=jnp.bool_)
        return_so_far = jnp.zeros((num_episodes,), dtype=jnp.float32)
        prev_true_recipe = initial_recipe.astype(jnp.int32)
        pair_key = jax.random.PRNGKey(seed)
        pair_key = jax.random.fold_in(pair_key, fcp_run_id)
        pair_key = jax.random.fold_in(pair_key, partner_run_id)

        def _step(carry, _):
            (
                obs,
                state,
                partner_h,
                fcp_h,
                ctx_state,
                done,
                return_so_far,
                prev_true_recipe,
                pair_key,
            ) = carry

            alive = ~done
            true_recipe_code = _recipe_bits_from_state(state).astype(jnp.int32)
            true_recipe = _recipe_labels(true_recipe_code, recipe_codes).astype(jnp.int32)
            valid = ctx_state.valid_mask & alive
            pred_recipe = jnp.where(valid, jnp.argmax(ctx_state.recipe_ctx, axis=-1), -1)
            pred_recipe_clipped = jnp.clip(pred_recipe, 0, len(recipe_codes) - 1)
            pred_recipe_code = jnp.where(valid, recipe_codes[pred_recipe_clipped], -1)
            max_prob = jnp.where(valid, jnp.max(ctx_state.recipe_ctx, axis=-1), jnp.nan)
            prob_true = jnp.take_along_axis(
                ctx_state.recipe_ctx,
                true_recipe[:, None],
                axis=-1,
            ).squeeze(axis=-1)
            prob_true = jnp.where(valid, prob_true, jnp.nan)
            correct = valid & (pred_recipe == true_recipe)
            if len(recipe_codes) == 2:
                swapped_recipe = jnp.where(valid, 1 - pred_recipe, -1)
                swapped_correct = valid & (swapped_recipe == true_recipe)
            else:
                swapped_correct = jnp.zeros_like(correct, dtype=jnp.bool_)
            recipe_changed = alive & (true_recipe != prev_true_recipe)

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
            next_obs, next_state, reward, dones, _ = env_step_vmapped(step_keys, state, actions)

            active_reward = reward["agent_1"] * alive.astype(reward["agent_1"].dtype)
            next_return_so_far = return_so_far + active_reward

            next_recipe = _recipe_labels(_recipe_bits_from_state(next_state), recipe_codes)
            next_ctx_state = context_manager.update(
                state=ctx_state,
                ego_obs=jnp.asarray(obs["agent_1"]),
                partner_act=jnp.asarray(a0, dtype=jnp.int32),
                current_recipes=jnp.asarray(true_recipe, dtype=jnp.int32),
                dones=jnp.asarray(dones["__all__"], dtype=jnp.bool_),
                next_recipes=jnp.asarray(next_recipe, dtype=jnp.int32),
            )
            next_done = done | jnp.asarray(dones["__all__"], dtype=jnp.bool_)

            next_carry = (
                next_obs,
                next_state,
                partner_h,
                fcp_h,
                next_ctx_state,
                next_done,
                next_return_so_far,
                true_recipe,
                pair_key,
            )
            records = {
                "valid": valid,
                "true_recipe": true_recipe,
                "true_recipe_code": true_recipe_code,
                "pred_recipe": pred_recipe,
                "pred_recipe_code": pred_recipe_code,
                "correct": correct,
                "swapped_correct": swapped_correct,
                "recipe_changed": recipe_changed,
                "max_prob": max_prob,
                "prob_true_recipe": prob_true,
                "episode_alive": alive,
                "reward_t": active_reward,
                "return_so_far": next_return_so_far,
            }
            return next_carry, records

        carry0 = (
            obs,
            state,
            partner_h0,
            fcp_h0,
            ctx_state,
            done,
            return_so_far,
            prev_true_recipe,
            pair_key,
        )
        _, arrays = jax.lax.scan(_step, carry0, None, length=episode_limit)
        return arrays

    return jax.jit(_rollout)


def _run_pair_online_accuracy_scanned(
    scanned_rollout_fn: Callable,
    fcp_policy: ContextualPPOPolicy,
    partner_policy: ContextualPPOPolicy,
    fcp_run_id: int,
    partner_run_id: int,
    num_episodes: int,
    seed: int,
    k: int,
    recipe_codes,
) -> PairRollout:
    partner_h0 = partner_policy.init_hstate(batch_size=num_episodes)
    fcp_h0 = fcp_policy.init_hstate(batch_size=num_episodes)
    arrays = scanned_rollout_fn(
        partner_policy.params,
        fcp_policy.params,
        partner_h0,
        fcp_h0,
        jnp.asarray(fcp_run_id, dtype=jnp.uint32),
        jnp.asarray(partner_run_id, dtype=jnp.uint32),
        jnp.asarray(seed, dtype=jnp.uint32),
        recipe_codes,
    )
    arrays = jax.tree_util.tree_map(np.asarray, arrays)

    any_alive = np.any(arrays["episode_alive"].astype(bool), axis=1)
    stopped = np.where(~any_alive)[0]
    if len(stopped) > 0:
        num_steps = int(stopped[0])
        arrays = {name: values[:num_steps] for name, values in arrays.items()}

    return _pair_rollout_from_arrays(
        arrays=arrays,
        k=k,
        fcp_run_id=fcp_run_id,
        partner_run_id=partner_run_id,
        num_episodes=num_episodes,
    )


def _run_pair_online_accuracy(
    env_reset_vmapped,
    env_step_vmapped,
    fcp_policy: ContextualPPOPolicy,
    partner_policy: ContextualPPOPolicy,
    context_manager: RecipeContextManager,
    fcp_run_id: int,
    partner_run_id: int,
    num_episodes: int,
    seed: int,
    k: int,
    recipe_codes,
    env_max_steps: int,
    max_steps: Optional[int] = None,
) -> PairRollout:
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
    return_so_far = jnp.zeros((num_episodes,), dtype=jnp.float32)
    prev_true_recipe = initial_recipe.astype(jnp.int32)
    pair_key = jax.random.PRNGKey(seed)
    pair_key = jax.random.fold_in(pair_key, int(fcp_run_id))
    pair_key = jax.random.fold_in(pair_key, int(partner_run_id))

    timestep_records: Dict[str, List[np.ndarray]] = {
        "valid": [],
        "true_recipe": [],
        "true_recipe_code": [],
        "pred_recipe": [],
        "pred_recipe_code": [],
        "correct": [],
        "swapped_correct": [],
        "recipe_changed": [],
        "max_prob": [],
        "prob_true_recipe": [],
        "episode_alive": [],
        "reward_t": [],
        "return_so_far": [],
    }

    for _t in range(episode_limit):
        if bool(jnp.all(done)):
            break

        alive = ~done
        true_recipe_code = _recipe_bits_from_state(state).astype(jnp.int32)
        true_recipe = _recipe_labels(true_recipe_code, recipe_codes).astype(jnp.int32)
        valid = ctx_state.valid_mask & alive
        pred_recipe = jnp.where(valid, jnp.argmax(ctx_state.recipe_ctx, axis=-1), -1)
        pred_recipe_clipped = jnp.clip(pred_recipe, 0, len(recipe_codes) - 1)
        pred_recipe_code = jnp.where(valid, recipe_codes[pred_recipe_clipped], -1)
        max_prob = jnp.where(valid, jnp.max(ctx_state.recipe_ctx, axis=-1), jnp.nan)
        prob_true = jnp.take_along_axis(
            ctx_state.recipe_ctx,
            true_recipe[:, None],
            axis=-1,
        ).squeeze(axis=-1)
        prob_true = jnp.where(valid, prob_true, jnp.nan)
        correct = valid & (pred_recipe == true_recipe)
        if len(recipe_codes) == 2:
            swapped_recipe = jnp.where(valid, 1 - pred_recipe, -1)
            swapped_correct = valid & (swapped_recipe == true_recipe)
        else:
            swapped_correct = jnp.zeros_like(correct, dtype=jnp.bool_)
        recipe_changed = alive & (true_recipe != prev_true_recipe)

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
        next_obs, next_state, reward, dones, _ = env_step_vmapped(step_keys, state, actions)

        active_reward = reward["agent_1"] * alive.astype(reward["agent_1"].dtype)
        return_so_far = return_so_far + active_reward

        timestep_records["valid"].append(np.asarray(valid))
        timestep_records["true_recipe"].append(np.asarray(true_recipe))
        timestep_records["true_recipe_code"].append(np.asarray(true_recipe_code))
        timestep_records["pred_recipe"].append(np.asarray(pred_recipe))
        timestep_records["pred_recipe_code"].append(np.asarray(pred_recipe_code))
        timestep_records["correct"].append(np.asarray(correct))
        timestep_records["swapped_correct"].append(np.asarray(swapped_correct))
        timestep_records["recipe_changed"].append(np.asarray(recipe_changed))
        timestep_records["max_prob"].append(np.asarray(max_prob))
        timestep_records["prob_true_recipe"].append(np.asarray(prob_true))
        timestep_records["episode_alive"].append(np.asarray(alive))
        timestep_records["reward_t"].append(np.asarray(active_reward))
        timestep_records["return_so_far"].append(np.asarray(return_so_far))

        next_recipe = _recipe_labels(_recipe_bits_from_state(next_state), recipe_codes)
        ctx_state = context_manager.update(
            state=ctx_state,
            ego_obs=jnp.asarray(obs["agent_1"]),
            partner_act=jnp.asarray(a0, dtype=jnp.int32),
            current_recipes=jnp.asarray(true_recipe, dtype=jnp.int32),
            dones=jnp.asarray(dones["__all__"], dtype=jnp.bool_),
            next_recipes=jnp.asarray(next_recipe, dtype=jnp.int32),
        )

        prev_true_recipe = true_recipe
        obs, state = next_obs, next_state
        done = done | jnp.asarray(dones["__all__"], dtype=jnp.bool_)

    arrays = {name: np.stack(values, axis=0) for name, values in timestep_records.items()}
    return _pair_rollout_from_arrays(
        arrays=arrays,
        k=k,
        fcp_run_id=fcp_run_id,
        partner_run_id=partner_run_id,
        num_episodes=num_episodes,
    )


def _write_rows(path: Path, rows: List[dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def _append_rows(path: Path, rows: List[dict], fieldnames: List[str]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists()) or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()
        os.fsync(f.fileno())


def _pair_string(pair: Tuple[int, int]) -> str:
    return f"{int(pair[0])}:{int(pair[1])}"


def _parse_pair_string(value: str) -> Tuple[int, int]:
    fcp_run, partner_run = value.split(":", 1)
    return int(fcp_run), int(partner_run)


def _safe_int(value, default: int = -1) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _load_completed_pairs_from_episode_csv(
    path: Path,
    k: int,
    episodes_per_pair: int,
) -> Set[Tuple[int, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()

    episodes_by_pair: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"k", "fcp_run", "partner_run", "episode"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                return set()
            for row in reader:
                if _safe_int(row.get("k")) != int(k):
                    continue
                pair = (_safe_int(row.get("fcp_run")), _safe_int(row.get("partner_run")))
                episode = _safe_int(row.get("episode"))
                if pair[0] < 0 or pair[1] < 0 or episode < 0:
                    continue
                episodes_by_pair[pair].add(episode)
    except Exception:
        return set()

    expected_episodes = set(range(int(episodes_per_pair)))
    return {
        pair
        for pair, episodes in episodes_by_pair.items()
        if expected_episodes.issubset(episodes)
    }


def _load_pairs_with_timestep_rows(path: Path, k: int) -> Set[Tuple[int, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()

    pairs = set()
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"k", "fcp_run", "partner_run"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                return set()
            for row in reader:
                if _safe_int(row.get("k")) != int(k):
                    continue
                pair = (_safe_int(row.get("fcp_run")), _safe_int(row.get("partner_run")))
                if pair[0] >= 0 and pair[1] >= 0:
                    pairs.add(pair)
    except Exception:
        return set()
    return pairs


def _load_resume_rows(
    path: Path,
    fieldnames: List[str],
    completed_pairs: Set[Tuple[int, int]],
    unique_columns: Tuple[str, ...],
) -> List[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []

    rows = []
    seen = set()
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"fcp_run", "partner_run", *unique_columns}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                return []
            for row in reader:
                pair = (_safe_int(row.get("fcp_run")), _safe_int(row.get("partner_run")))
                if pair not in completed_pairs:
                    continue
                unique_key = (
                    pair,
                    *(_safe_int(row.get(col)) for col in unique_columns),
                )
                if unique_key in seen:
                    continue
                seen.add(unique_key)
                rows.append({col: row.get(col, "") for col in fieldnames})
    except Exception:
        return []
    return rows


def _validate_existing_csv_schema(path: Path, fieldnames: List[str]):
    if not path.exists() or path.stat().st_size == 0:
        return

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        existing = set(reader.fieldnames or [])

    missing = [name for name in fieldnames if name not in existing]
    if missing:
        raise ValueError(
            f"Existing CSV schema is missing columns {missing}: {path}. "
            "Use --overwrite_output to regenerate it or write to a new output path."
        )


def _progress_settings(
    args,
    encoder_k: int,
    expected_max_steps: int,
    env_max_steps: int,
    sp_run_ids: List[int],
    fcp_run_ids: List[int],
) -> dict:
    return {
        "version": 2,
        "layout": str(args.layout),
        "agent_view_size": int(args.agent_view_size),
        "episodes_per_pair": int(args.episodes_per_pair),
        "max_steps": int(expected_max_steps),
        "env_max_steps": int(env_max_steps),
        "seed": int(args.seed),
        "encoder_k": int(encoder_k),
        "stochastic": bool(args.stochastic),
        "sp_dir": str(args.sp_dir),
        "fcp_encoder_dir": str(args.fcp_encoder_dir),
        "encoder_ckpt": str(args.encoder_ckpt),
        "sp_run_ids": [int(x) for x in sp_run_ids],
        "fcp_run_ids": [int(x) for x in fcp_run_ids],
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
            f"Use --overwrite_output to start fresh, or write to new output paths. "
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


def _mean_metric(rows: List[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if np.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def _mean_nonnegative_metric(rows: List[dict], key: str) -> float:
    vals = [
        float(r[key])
        for r in rows
        if np.isfinite(float(r[key])) and float(r[key]) >= 0
    ]
    return float(np.mean(vals)) if vals else float("nan")


def _summary_row(k, metric, group, value, n, fcp=-1, partner=-1, recipe=-1, timestep=-1):
    return {
        "k": k,
        "metric": metric,
        "group": group,
        "fcp_run": fcp,
        "partner_run": partner,
        "recipe": recipe,
        "timestep": timestep,
        "value": value,
        "n": n,
    }


def _build_summary(k: int, timestep_rows: List[dict], episode_rows: List[dict]) -> List[dict]:
    summary = []
    alive_rows = [r for r in timestep_rows if int(r["episode_alive"]) == 1]
    valid_rows = [r for r in alive_rows if int(r["valid"]) == 1]
    correct_rows = [r for r in valid_rows if int(r["correct"]) == 1]
    swapped_correct_rows = [r for r in valid_rows if int(r.get("swapped_correct", 0)) == 1]

    overall_valid_acc = len(correct_rows) / len(valid_rows) if valid_rows else float("nan")
    swapped_valid_acc = (
        len(swapped_correct_rows) / len(valid_rows)
        if valid_rows
        else float("nan")
    )
    direct_minus_swapped = (
        overall_valid_acc - swapped_valid_acc
        if np.isfinite(overall_valid_acc) and np.isfinite(swapped_valid_acc)
        else float("nan")
    )
    effective_correct = len(correct_rows) / len(alive_rows) if alive_rows else float("nan")
    mean_coverage = len(valid_rows) / len(alive_rows) if alive_rows else float("nan")
    num_classes = len({int(r["true_recipe"]) for r in alive_rows}) if alive_rows else 0
    label_mismatch_suspected = (
        int(num_classes == 2 and np.isfinite(direct_minus_swapped) and direct_minus_swapped < -0.10)
    )

    after_change_rows = []
    rows_by_episode: Dict[Tuple[int, int, int], List[dict]] = defaultdict(list)
    for row in alive_rows:
        key = (int(row["fcp_run"]), int(row["partner_run"]), int(row["episode"]))
        rows_by_episode[key].append(row)

    for rows in rows_by_episode.values():
        seen_change = False
        for row in sorted(rows, key=lambda x: int(x["timestep"])):
            if int(row["recipe_changed"]) == 1:
                seen_change = True
            if seen_change and int(row["valid"]) == 1:
                after_change_rows.append(row)

    after_change_correct = [
        r for r in after_change_rows if int(r["correct"]) == 1
    ]
    after_change_acc = (
        len(after_change_correct) / len(after_change_rows)
        if after_change_rows
        else float("nan")
    )

    summary.extend(
        [
            _summary_row(k, "overall_valid_accuracy", "global", overall_valid_acc, len(valid_rows)),
            _summary_row(k, "swapped_valid_accuracy", "global", swapped_valid_acc, len(valid_rows)),
            _summary_row(k, "direct_minus_swapped_accuracy", "global", direct_minus_swapped, len(valid_rows)),
            _summary_row(k, "label_mismatch_suspected", "global", float(label_mismatch_suspected), len(valid_rows)),
            _summary_row(k, "accuracy_after_recipe_change", "global", after_change_acc, len(after_change_rows)),
            _summary_row(k, "overall_effective_correct_rate", "global", effective_correct, len(alive_rows)),
            _summary_row(k, "mean_coverage", "global", mean_coverage, len(alive_rows)),
            _summary_row(k, "mean_first_valid_timestep", "global", _mean_nonnegative_metric(episode_rows, "first_valid_timestep"), len(episode_rows)),
            _summary_row(k, "mean_first_correct_timestep", "global", _mean_nonnegative_metric(episode_rows, "first_correct_timestep"), len(episode_rows)),
            _summary_row(k, "mean_flip_rate", "global", _mean_metric(episode_rows, "flip_rate"), len(episode_rows)),
            _summary_row(k, "mean_prob_true_recipe", "global", _mean_metric(episode_rows, "mean_prob_true_recipe"), len(episode_rows)),
            _summary_row(k, "mean_episode_return", "global", _mean_metric(episode_rows, "episode_return"), len(episode_rows)),
        ]
    )

    timesteps = sorted({int(r["timestep"]) for r in timestep_rows})
    for t in timesteps:
        at_t = [r for r in timestep_rows if int(r["timestep"]) == t and int(r["episode_alive"]) == 1]
        valid_t = [r for r in at_t if int(r["valid"]) == 1]
        correct_t = [r for r in valid_t if int(r["correct"]) == 1]
        n_alive = len(at_t)
        summary.append(_summary_row(k, "accuracy_by_timestep", "timestep", len(correct_t) / len(valid_t) if valid_t else float("nan"), len(valid_t), timestep=t))
        summary.append(_summary_row(k, "coverage_by_timestep", "timestep", len(valid_t) / n_alive if n_alive else float("nan"), n_alive, timestep=t))
        summary.append(_summary_row(k, "valid_and_correct_by_timestep", "timestep", len(correct_t) / n_alive if n_alive else float("nan"), n_alive, timestep=t))

    for partner in sorted({int(r["partner_run"]) for r in timestep_rows}):
        rows = [r for r in valid_rows if int(r["partner_run"]) == partner]
        correct = [r for r in rows if int(r["correct"]) == 1]
        summary.append(_summary_row(k, "accuracy_by_partner", "partner", len(correct) / len(rows) if rows else float("nan"), len(rows), partner=partner))

    for fcp in sorted({int(r["fcp_run"]) for r in timestep_rows}):
        rows = [r for r in valid_rows if int(r["fcp_run"]) == fcp]
        correct = [r for r in rows if int(r["correct"]) == 1]
        summary.append(_summary_row(k, "accuracy_by_fcp", "fcp", len(correct) / len(rows) if rows else float("nan"), len(rows), fcp=fcp))

    pairs = sorted({(int(r["fcp_run"]), int(r["partner_run"])) for r in timestep_rows})
    for fcp, partner in pairs:
        rows = [r for r in valid_rows if int(r["fcp_run"]) == fcp and int(r["partner_run"]) == partner]
        correct = [r for r in rows if int(r["correct"]) == 1]
        returns = [
            float(r["episode_return"])
            for r in episode_rows
            if int(r["fcp_run"]) == fcp and int(r["partner_run"]) == partner
        ]
        summary.append(_summary_row(k, "accuracy_by_pair", "pair", len(correct) / len(rows) if rows else float("nan"), len(rows), fcp=fcp, partner=partner))
        summary.append(_summary_row(k, "return_by_pair", "pair", float(np.mean(returns)) if returns else float("nan"), len(returns), fcp=fcp, partner=partner))

    for recipe in sorted({int(r["true_recipe"]) for r in timestep_rows}):
        rows = [r for r in valid_rows if int(r["true_recipe"]) == recipe]
        correct = [r for r in rows if int(r["correct"]) == 1]
        summary.append(_summary_row(k, "accuracy_by_recipe", "recipe", len(correct) / len(rows) if rows else float("nan"), len(rows), recipe=recipe))

    return summary


def _build_summary_from_csv(k: int, timestep_csv: Path, episode_csv: Path) -> List[dict]:
    alive_count = 0
    valid_count = 0
    correct_count = 0
    swapped_correct_count = 0
    after_change_valid_count = 0
    after_change_correct_count = 0
    seen_change = set()
    seen_recipes = set()

    timestep_counts = defaultdict(lambda: [0, 0, 0])  # alive, valid, correct
    partner_counts = defaultdict(lambda: [0, 0])  # valid, correct
    fcp_counts = defaultdict(lambda: [0, 0])
    pair_counts = defaultdict(lambda: [0, 0])
    recipe_counts = defaultdict(lambda: [0, 0])

    def _as_int(row, key, default=0):
        try:
            return int(float(row.get(key, default)))
        except Exception:
            return default

    def _as_float(row, key):
        try:
            return float(row.get(key, "nan"))
        except Exception:
            return float("nan")

    with open(timestep_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alive = _as_int(row, "episode_alive") == 1
            if not alive:
                continue

            fcp = _as_int(row, "fcp_run")
            partner = _as_int(row, "partner_run")
            episode = _as_int(row, "episode")
            timestep = _as_int(row, "timestep")
            recipe = _as_int(row, "true_recipe")
            valid = _as_int(row, "valid") == 1
            correct = _as_int(row, "correct") == 1
            swapped_correct = _as_int(row, "swapped_correct") == 1
            recipe_changed = _as_int(row, "recipe_changed") == 1

            key = (fcp, partner, episode)
            if recipe_changed:
                seen_change.add(key)

            seen_recipes.add(recipe)
            alive_count += 1
            timestep_counts[timestep][0] += 1

            if not valid:
                continue

            valid_count += 1
            timestep_counts[timestep][1] += 1
            partner_counts[partner][0] += 1
            fcp_counts[fcp][0] += 1
            pair_counts[(fcp, partner)][0] += 1
            recipe_counts[recipe][0] += 1

            if correct:
                correct_count += 1
                timestep_counts[timestep][2] += 1
                partner_counts[partner][1] += 1
                fcp_counts[fcp][1] += 1
                pair_counts[(fcp, partner)][1] += 1
                recipe_counts[recipe][1] += 1

            if swapped_correct:
                swapped_correct_count += 1

            if key in seen_change:
                after_change_valid_count += 1
                if correct:
                    after_change_correct_count += 1

    episode_count = 0
    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)
    nonneg_sums = defaultdict(float)
    nonneg_counts = defaultdict(int)
    return_by_pair = defaultdict(lambda: [0.0, 0])

    with open(episode_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episode_count += 1
            fcp = _as_int(row, "fcp_run")
            partner = _as_int(row, "partner_run")
            episode_return = _as_float(row, "episode_return")
            if math.isfinite(episode_return):
                return_by_pair[(fcp, partner)][0] += episode_return
                return_by_pair[(fcp, partner)][1] += 1

            for key in [
                "mean_flip_rate",
                "mean_prob_true_recipe",
                "episode_return",
            ]:
                source_key = "flip_rate" if key == "mean_flip_rate" else (
                    "mean_prob_true_recipe" if key == "mean_prob_true_recipe" else "episode_return"
                )
                value = _as_float(row, source_key)
                if math.isfinite(value):
                    metric_sums[key] += value
                    metric_counts[key] += 1

            for key, source_key in [
                ("mean_first_valid_timestep", "first_valid_timestep"),
                ("mean_first_correct_timestep", "first_correct_timestep"),
            ]:
                value = _as_float(row, source_key)
                if math.isfinite(value) and value >= 0:
                    nonneg_sums[key] += value
                    nonneg_counts[key] += 1

    summary = []
    overall_valid_acc = correct_count / valid_count if valid_count else float("nan")
    swapped_valid_acc = swapped_correct_count / valid_count if valid_count else float("nan")
    direct_minus_swapped = (
        overall_valid_acc - swapped_valid_acc
        if math.isfinite(overall_valid_acc) and math.isfinite(swapped_valid_acc)
        else float("nan")
    )
    effective_correct = correct_count / alive_count if alive_count else float("nan")
    mean_coverage = valid_count / alive_count if alive_count else float("nan")
    label_mismatch_suspected = int(
        len(seen_recipes) == 2
        and math.isfinite(direct_minus_swapped)
        and direct_minus_swapped < -0.10
    )
    after_change_acc = (
        after_change_correct_count / after_change_valid_count
        if after_change_valid_count
        else float("nan")
    )

    summary.extend(
        [
            _summary_row(k, "overall_valid_accuracy", "global", overall_valid_acc, valid_count),
            _summary_row(k, "swapped_valid_accuracy", "global", swapped_valid_acc, valid_count),
            _summary_row(k, "direct_minus_swapped_accuracy", "global", direct_minus_swapped, valid_count),
            _summary_row(k, "label_mismatch_suspected", "global", float(label_mismatch_suspected), valid_count),
            _summary_row(k, "accuracy_after_recipe_change", "global", after_change_acc, after_change_valid_count),
            _summary_row(k, "overall_effective_correct_rate", "global", effective_correct, alive_count),
            _summary_row(k, "mean_coverage", "global", mean_coverage, alive_count),
            _summary_row(
                k,
                "mean_first_valid_timestep",
                "global",
                nonneg_sums["mean_first_valid_timestep"] / nonneg_counts["mean_first_valid_timestep"]
                if nonneg_counts["mean_first_valid_timestep"]
                else float("nan"),
                episode_count,
            ),
            _summary_row(
                k,
                "mean_first_correct_timestep",
                "global",
                nonneg_sums["mean_first_correct_timestep"] / nonneg_counts["mean_first_correct_timestep"]
                if nonneg_counts["mean_first_correct_timestep"]
                else float("nan"),
                episode_count,
            ),
            _summary_row(
                k,
                "mean_flip_rate",
                "global",
                metric_sums["mean_flip_rate"] / metric_counts["mean_flip_rate"]
                if metric_counts["mean_flip_rate"]
                else float("nan"),
                episode_count,
            ),
            _summary_row(
                k,
                "mean_prob_true_recipe",
                "global",
                metric_sums["mean_prob_true_recipe"] / metric_counts["mean_prob_true_recipe"]
                if metric_counts["mean_prob_true_recipe"]
                else float("nan"),
                episode_count,
            ),
            _summary_row(
                k,
                "mean_episode_return",
                "global",
                metric_sums["episode_return"] / metric_counts["episode_return"]
                if metric_counts["episode_return"]
                else float("nan"),
                episode_count,
            ),
        ]
    )

    for timestep in sorted(timestep_counts):
        n_alive, n_valid, n_correct = timestep_counts[timestep]
        summary.append(_summary_row(k, "accuracy_by_timestep", "timestep", n_correct / n_valid if n_valid else float("nan"), n_valid, timestep=timestep))
        summary.append(_summary_row(k, "coverage_by_timestep", "timestep", n_valid / n_alive if n_alive else float("nan"), n_alive, timestep=timestep))
        summary.append(_summary_row(k, "valid_and_correct_by_timestep", "timestep", n_correct / n_alive if n_alive else float("nan"), n_alive, timestep=timestep))

    for partner in sorted(partner_counts):
        n_valid, n_correct = partner_counts[partner]
        summary.append(_summary_row(k, "accuracy_by_partner", "partner", n_correct / n_valid if n_valid else float("nan"), n_valid, partner=partner))

    for fcp in sorted(fcp_counts):
        n_valid, n_correct = fcp_counts[fcp]
        summary.append(_summary_row(k, "accuracy_by_fcp", "fcp", n_correct / n_valid if n_valid else float("nan"), n_valid, fcp=fcp))

    for fcp, partner in sorted(pair_counts):
        n_valid, n_correct = pair_counts[(fcp, partner)]
        return_sum, return_count = return_by_pair[(fcp, partner)]
        summary.append(_summary_row(k, "accuracy_by_pair", "pair", n_correct / n_valid if n_valid else float("nan"), n_valid, fcp=fcp, partner=partner))
        summary.append(_summary_row(k, "return_by_pair", "pair", return_sum / return_count if return_count else float("nan"), return_count, fcp=fcp, partner=partner))

    for recipe in sorted(recipe_counts):
        n_valid, n_correct = recipe_counts[recipe]
        summary.append(_summary_row(k, "accuracy_by_recipe", "recipe", n_correct / n_valid if n_valid else float("nan"), n_valid, recipe=recipe))

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp_dir", required=True, type=Path)
    parser.add_argument("--fcp_encoder_dir", required=True, type=Path)
    parser.add_argument("--encoder_ckpt", required=True, type=Path)
    parser.add_argument("--layout", default="demo_cook_simple")
    parser.add_argument("--agent_view_size", type=int, default=2)
    parser.add_argument("--episodes_per_pair", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument(
        "--env_max_steps",
        type=int,
        default=None,
        help=(
            "Episode horizon passed to OvercookedV2. Defaults to max(--max_steps, 400), "
            "so long-horizon evals like --max_steps 2000 do not stop at the env default 400."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder_k", type=int, default=None)
    parser.add_argument("--output_timestep_csv", required=True, type=Path)
    parser.add_argument("--output_episode_csv", required=True, type=Path)
    parser.add_argument("--output_summary_csv", required=True, type=Path)
    parser.add_argument(
        "--progress_json",
        default=None,
        type=Path,
        help="Optional sidecar progress file. Defaults to <output_episode_csv>.progress.json.",
    )
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="Delete existing output/progress files and start from the first pair.",
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--disable_scanned_rollout",
        action="store_true",
        help="Use the original Python timestep loop instead of the jitted lax.scan rollout.",
    )
    args = parser.parse_args()

    env_horizon = int(args.env_max_steps) if args.env_max_steps is not None else max(int(args.max_steps), 400)
    if env_horizon < int(args.max_steps):
        raise ValueError(
            f"--env_max_steps ({env_horizon}) must be >= --max_steps ({args.max_steps})."
        )

    env_kwargs = {
        "layout": args.layout,
        "agent_view_size": args.agent_view_size,
        "random_agent_positions": False,
        "sample_recipe_on_delivery": True,
        "negative_rewards": True,
        "max_steps": env_horizon,
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
    recipe_codes = jnp.array(recipe_codes_np, dtype=jnp.int32)
    num_classes = int(len(recipe_codes))
    encoder_k = _resolve_encoder_k(encoder_meta_path, args.encoder_k)
    encoder_use_actions = bool(encoder_meta.get("use_actions", False))
    encoder_action_dim = int(encoder_meta.get("action_dim", 0))

    print(f"Recipe codebook used for labels: {recipe_codes_np.tolist()}")

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

    sp_run_ids = _list_run_ids(args.sp_dir)
    fcp_run_ids = _list_run_ids(args.fcp_encoder_dir)
    if not sp_run_ids:
        raise ValueError(f"No SP run_*/ckpt_final found in {args.sp_dir}")
    if not fcp_run_ids:
        raise ValueError(f"No FCP run_*/ckpt_final found in {args.fcp_encoder_dir}")

    expected_pairs = {
        (int(fcp_run_id), int(partner_run_id))
        for fcp_run_id in fcp_run_ids
        for partner_run_id in sp_run_ids
    }
    expected_max_steps = int(args.max_steps) if args.max_steps is not None else int(env.max_steps)
    progress_path = args.progress_json or Path(str(args.output_episode_csv) + ".progress.json")
    progress_settings = _progress_settings(
        args=args,
        encoder_k=encoder_k,
        expected_max_steps=expected_max_steps,
        env_max_steps=int(env.max_steps),
        sp_run_ids=sp_run_ids,
        fcp_run_ids=fcp_run_ids,
    )

    for output_path in [
        args.output_timestep_csv,
        args.output_episode_csv,
        args.output_summary_csv,
        progress_path,
    ]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite_output and output_path.exists():
            output_path.unlink()

    _validate_existing_csv_schema(args.output_timestep_csv, TIMESTEP_COLUMNS)
    _validate_existing_csv_schema(args.output_episode_csv, EPISODE_COLUMNS)
    _validate_existing_csv_schema(args.output_summary_csv, SUMMARY_COLUMNS)

    progress_completed = _load_progress(progress_path, progress_settings)
    csv_completed = _load_completed_pairs_from_episode_csv(
        args.output_episode_csv,
        k=encoder_k,
        episodes_per_pair=args.episodes_per_pair,
    )
    timestep_pairs = _load_pairs_with_timestep_rows(args.output_timestep_csv, k=encoder_k)

    if progress_completed is None:
        completed_pairs = csv_completed
    else:
        # Use the union so a VM preemption after CSV append but before progress
        # update still resumes from the completed CSV rows.
        completed_pairs = progress_completed | csv_completed

    completed_pairs = completed_pairs & timestep_pairs & expected_pairs

    timestep_rows = _load_resume_rows(
        args.output_timestep_csv,
        TIMESTEP_COLUMNS,
        completed_pairs,
        unique_columns=("episode", "timestep"),
    )
    episode_rows = _load_resume_rows(
        args.output_episode_csv,
        EPISODE_COLUMNS,
        completed_pairs,
        unique_columns=("episode",),
    )

    # Clean up stale partial rows before continuing. This prevents duplicate rows
    # if the VM was preempted after appending a pair but before progress was saved.
    _write_rows(args.output_timestep_csv, timestep_rows, TIMESTEP_COLUMNS)
    _write_rows(args.output_episode_csv, episode_rows, EPISODE_COLUMNS)
    _write_progress(progress_path, progress_settings, completed_pairs)
    # From here on, keep raw rollout rows on disk only. This lets larger
    # episode/timestep sweeps avoid carrying the full timestep CSV in memory.
    del timestep_rows
    del episode_rows

    print(
        "[Resume] completed pairs: "
        f"{len(completed_pairs)}/{len(expected_pairs)} "
        f"(progress: {progress_path})"
    )
    print(f"Environment max_steps: {int(env.max_steps)}; eval rollout steps: {expected_max_steps}")

    sp_policy_cache = {
        run_id: _load_policy(args.sp_dir, run_id, stochastic=args.stochastic)
        for run_id in sp_run_ids
    }
    fcp_policy_cache = {
        run_id: _load_policy(args.fcp_encoder_dir, run_id, stochastic=args.stochastic)
        for run_id in fcp_run_ids
    }

    scanned_rollout_fn = None
    if not args.disable_scanned_rollout:
        scanned_rollout_fn = _make_scanned_online_accuracy_rollout(
            env_reset_vmapped=env_reset_vmapped,
            env_step_vmapped=env_step_vmapped,
            fcp_policy=fcp_policy_cache[fcp_run_ids[0]],
            partner_policy=sp_policy_cache[sp_run_ids[0]],
            context_manager=context_manager,
            num_episodes=args.episodes_per_pair,
            episode_limit=expected_max_steps,
            stochastic=args.stochastic,
        )
        print(
            "[Fast rollout] using jitted lax.scan over "
            f"{expected_max_steps} timesteps; first new pair includes compile time."
        )

    pbar = tqdm(total=len(fcp_run_ids) * len(sp_run_ids), desc="online recipe accuracy pairs")
    pbar.update(len(completed_pairs))
    for fcp_idx, fcp_run_id in enumerate(fcp_run_ids):
        for partner_idx, partner_run_id in enumerate(sp_run_ids):
            pair_order = fcp_idx * len(sp_run_ids) + partner_idx + 1
            pair = (int(fcp_run_id), int(partner_run_id))

            if pair in completed_pairs:
                pbar.set_postfix(
                    {
                        "status": "resume-skip",
                        "pair": pair_order,
                        "fcp": fcp_run_id,
                        "sp": partner_run_id,
                    }
                )
                continue

            if scanned_rollout_fn is None:
                rollout = _run_pair_online_accuracy(
                    env_reset_vmapped=env_reset_vmapped,
                    env_step_vmapped=env_step_vmapped,
                    fcp_policy=fcp_policy_cache[fcp_run_id],
                    partner_policy=sp_policy_cache[partner_run_id],
                    context_manager=context_manager,
                    fcp_run_id=fcp_run_id,
                    partner_run_id=partner_run_id,
                    num_episodes=args.episodes_per_pair,
                    seed=args.seed,
                    k=encoder_k,
                    recipe_codes=recipe_codes,
                    env_max_steps=int(env.max_steps),
                    max_steps=args.max_steps,
                )
            else:
                rollout = _run_pair_online_accuracy_scanned(
                    scanned_rollout_fn=scanned_rollout_fn,
                    fcp_policy=fcp_policy_cache[fcp_run_id],
                    partner_policy=sp_policy_cache[partner_run_id],
                    fcp_run_id=fcp_run_id,
                    partner_run_id=partner_run_id,
                    num_episodes=args.episodes_per_pair,
                    seed=args.seed,
                    k=encoder_k,
                    recipe_codes=recipe_codes,
            )
            _append_rows(args.output_timestep_csv, rollout.timestep_rows, TIMESTEP_COLUMNS)
            _append_rows(args.output_episode_csv, rollout.episode_rows, EPISODE_COLUMNS)
            completed_pairs.add(pair)
            _write_progress(progress_path, progress_settings, completed_pairs)

            pbar.update(1)
            pbar.set_postfix(
                {
                    "status": "saved",
                    "pair": pair_order,
                    "fcp": fcp_run_id,
                    "sp": partner_run_id,
                }
            )
    pbar.close()

    summary_rows = _build_summary_from_csv(
        encoder_k,
        args.output_timestep_csv,
        args.output_episode_csv,
    )
    _write_rows(args.output_summary_csv, summary_rows, SUMMARY_COLUMNS)

    print(f"Saved timestep csv: {args.output_timestep_csv.resolve()}")
    print(f"Saved episode csv:  {args.output_episode_csv.resolve()}")
    print(f"Saved summary csv:  {args.output_summary_csv.resolve()}")
    print(f"Saved progress json: {progress_path.resolve()}")


if __name__ == "__main__":
    main()
