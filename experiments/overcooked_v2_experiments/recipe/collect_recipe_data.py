import json
import os
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from jaxmarl.environments.overcooked_v2.common import Actions, DynamicObject
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from overcooked_v2_experiments.ppo.policy import PPOPolicy
from overcooked_v2_experiments.recipe.masking import get_mask_fn


def load_partner_params(path):
    """Load PPO params from checkpoint."""
    ckpt_dir = Path(path)
    if ckpt_dir.name == "checkpoint":
        ckpt_dir = ckpt_dir.parent

    orbax_checkpointer = ocp.PyTreeCheckpointer()
    try:
        ckpt = orbax_checkpointer.restore(ckpt_dir, item=None)
        if "params" in ckpt:
            params = ckpt["params"]
            config = ckpt.get("config", {})
        else:
            params = ckpt
            config = {}
        return params, config
    except Exception as e:
        print(f"Error loading policy from {path}: {e}")
        return None, None


def _recipe_code_mapping(env):
    recipe_candidates = jnp.array(env.layout.possible_recipes, dtype=jnp.int32)
    recipe_codes = jax.vmap(DynamicObject.get_recipe_encoding)(recipe_candidates)
    recipe_codes = np.array(recipe_codes, dtype=np.int32)
    code_to_label = {int(code): idx for idx, code in enumerate(recipe_codes.tolist())}
    return recipe_codes, code_to_label


def _partner_visible_in_obs(obs_step: np.ndarray, num_ingredients: int) -> bool:
    """Returns whether agent_1 can see agent_0 in current ego observation."""
    ingredient_layer_width = 2 + num_ingredients
    agent_layer_width = 1 + 4 + ingredient_layer_width
    other_agent_pos_channel = agent_layer_width
    return bool(np.any(obs_step[..., other_agent_pos_channel] > 0))


def _segment_partner_action(
    seg_obs,
    seg_act,
    num_ingredients,
    include_partner_action,
    visibility_aware_actions,
):
    if not include_partner_action:
        return None

    if not visibility_aware_actions:
        # Standard one-hot over 6 discrete actions.
        return np.eye(len(Actions), dtype=np.float32)[seg_act]

    # Visibility-aware action encoding with UNK bucket at last index.
    action_dim = len(Actions) + 1
    action_onehot = np.zeros((seg_act.shape[0], action_dim), dtype=np.float32)
    for t, action_id in enumerate(seg_act.tolist()):
        if _partner_visible_in_obs(seg_obs[t], num_ingredients):
            action_onehot[t, int(action_id)] = 1.0
        else:
            action_onehot[t, -1] = 1.0  # UNK
    return action_onehot


@hydra.main(version_base=None, config_path="../ppo/config", config_name="base")
def main(config: DictConfig):
    print(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    sp_run_dir = config.get("SP_RUN_DIR", None)
    save_dir = config.get("SAVE_DIR", "./runs/recipe_data_demo_cook_simple_obs_only")

    n_episodes = int(config.get("RECIPE_EPISODES_PER_PARTNER", 50))
    max_steps = int(config.get("RECIPE_MAX_STEPS", 400))
    segment_k = int(config.get("RECIPE_SEGMENT_K", 10))
    segment_stride = int(config.get("RECIPE_SEGMENT_STRIDE", segment_k))

    include_partner_action = bool(config.get("RECIPE_INCLUDE_PARTNER_ACTION", False))
    visibility_aware_actions = bool(
        config.get("RECIPE_PARTNER_ACTION_VISIBILITY_AWARE", False)
    )

    if not sp_run_dir:
        print("Error: SP_RUN_DIR not specified.")
        return

    sp_run_path = Path(sp_run_dir)
    if not sp_run_path.exists():
        print(f"Error: SP_RUN_DIR {sp_run_path} does not exist.")
        return

    # Find checkpoints (one partner policy per run directory).
    partner_ckpts = []
    run_dirs = sorted(
        [d for d in sp_run_path.iterdir() if d.is_dir() and d.name.startswith("run_")]
    )
    for run_dir in run_dirs:
        ckpt_path = run_dir / "ckpt_final"
        if ckpt_path.exists():
            partner_ckpts.append(str(ckpt_path))

    if not partner_ckpts:
        print("No checkpoints found!")
        return

    print(f"Found {len(partner_ckpts)} partners.")
    os.makedirs(save_dir, exist_ok=True)

    env_kwargs = OmegaConf.to_container(config.env.ENV_KWARGS, resolve=True)
    env = OvercookedV2(**env_kwargs)

    recipe_codes, code_to_label = _recipe_code_mapping(env)
    num_classes = len(recipe_codes)

    sample_params, sample_config = load_partner_params(partner_ckpts[0])
    if sample_params is None:
        return

    partner_policy = PPOPolicy(sample_params, sample_config, stochastic=True)

    mask_fn = get_mask_fn(num_ingredients=env.layout.num_ingredients)

    print("--- Dataset Target Definition ---")
    print("Encoder input observation: agent_1 (ego) only")
    print(f"Partial observability active: {env.agent_view_size is not None}")
    print(f"include_partner_action: {include_partner_action}")
    print(f"visibility_aware_actions: {visibility_aware_actions}")
    print(f"num_recipe_classes: {num_classes}")
    print(f"recipe_codes: {recipe_codes.tolist()}")
    print("--------------------------------")

    @partial(jax.jit, static_argnums=(2,))
    def run_rollout(rng, partner_params, num_steps=400):
        def _env_step(carry, _):
            env_state, last_obs, hstate, _, rng = carry
            rng, rng_act, rng_step = jax.random.split(rng, 3)

            partner_obs = last_obs["agent_0"]
            partner_action, new_hstate = partner_policy.compute_action(
                partner_obs,
                jnp.bool_(False),
                hstate,
                rng_act,
                params=partner_params,
            )

            ego_action = Actions.stay.value
            actions = {
                "agent_0": partner_action,
                "agent_1": ego_action,
            }

            new_obs, new_state, _, new_done, _ = env.step(rng_step, env_state, actions)

            ego_obs = mask_fn(last_obs["agent_1"])
            recipe_code = env_state.recipe
            done_signal = new_done["__all__"]

            step_data = (ego_obs, partner_action, recipe_code, done_signal)
            carry = (new_state, new_obs, new_hstate, done_signal, rng)
            return carry, step_data

        rng, rng_reset = jax.random.split(rng)
        obs, state = env.reset(rng_reset)
        hstate = partner_policy.init_hstate(batch_size=1)

        init_carry = (state, obs, hstate, jnp.bool_(False), rng)
        _, trajectory = jax.lax.scan(_env_step, init_carry, None, length=num_steps)
        return trajectory

    run_rollout_vmap = jax.vmap(run_rollout, in_axes=(0, None, None))

    total_samples = 0
    num_saved_files = 0

    pbar = tqdm(partner_ckpts, desc="Partners")
    for pid, ckpt_path in enumerate(pbar):
        params, _ = load_partner_params(ckpt_path)
        if params is None:
            continue

        master_key = jax.random.PRNGKey(pid * 1000)
        batch_keys = jax.random.split(master_key, n_episodes)

        traj_obs, traj_act, traj_rec, traj_done = run_rollout_vmap(
            batch_keys,
            params,
            max_steps,
        )

        traj_obs = np.array(traj_obs)
        traj_act = np.array(traj_act)
        traj_rec = np.array(traj_rec)
        traj_done = np.array(traj_done)

        obs_segments = []
        act_segments = []
        recipe_segments = []
        partner_segments = []

        for ep_idx in range(n_episodes):
            obs_seq = traj_obs[ep_idx]
            act_seq = traj_act[ep_idx]
            rec_seq = traj_rec[ep_idx]
            done_seq = traj_done[ep_idx]

            curr_start = 0
            while curr_start + segment_k <= max_steps:
                seg_rec = rec_seq[curr_start : curr_start + segment_k]
                seg_done = done_seq[curr_start : curr_start + segment_k]

                if np.any(seg_done):
                    curr_start += segment_stride
                    continue

                if not np.all(seg_rec == seg_rec[0]):
                    curr_start += 1
                    continue

                recipe_code = int(seg_rec[0])
                if recipe_code not in code_to_label:
                    curr_start += 1
                    continue

                seg_obs = obs_seq[curr_start : curr_start + segment_k]
                seg_act = act_seq[curr_start : curr_start + segment_k]

                encoded_action = _segment_partner_action(
                    seg_obs=seg_obs,
                    seg_act=seg_act,
                    num_ingredients=env.layout.num_ingredients,
                    include_partner_action=include_partner_action,
                    visibility_aware_actions=visibility_aware_actions,
                )

                obs_segments.append(seg_obs)
                if encoded_action is not None:
                    act_segments.append(encoded_action)
                recipe_segments.append(code_to_label[recipe_code])
                partner_segments.append(pid)

                curr_start += segment_stride

        if obs_segments:
            out_file = os.path.join(save_dir, f"partner_{pid}.npz")
            save_dict = {
                "obs": np.array(obs_segments, dtype=np.float32),
                "recipe": np.array(recipe_segments, dtype=np.int32),
                "partner": np.array(partner_segments, dtype=np.int32),
            }
            if include_partner_action:
                save_dict["act"] = np.array(act_segments, dtype=np.float32)
            np.savez(out_file, **save_dict)

            num_saved_files += 1
            total_samples += len(obs_segments)

    metadata = {
        "layout": env_kwargs.get("layout", "unknown"),
        "agent_view_size": env_kwargs.get("agent_view_size", None),
        "episodes_per_partner": n_episodes,
        "max_steps": max_steps,
        "segment_k": segment_k,
        "segment_stride": segment_stride,
        "num_partners": len(partner_ckpts),
        "num_saved_partner_files": num_saved_files,
        "num_total_segments": total_samples,
        "num_ingredients": int(env.layout.num_ingredients),
        "num_recipe_classes": int(num_classes),
        "recipe_codes": recipe_codes.tolist(),
        "recipe_include_partner_action": include_partner_action,
        "recipe_partner_action_visibility_aware": visibility_aware_actions,
        "recipe_action_dim": int(len(Actions) + 1)
        if include_partner_action and visibility_aware_actions
        else int(len(Actions) if include_partner_action else 0),
    }

    meta_path = Path(save_dir) / "dataset_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Data collection complete. Total segments: {total_samples}")
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
