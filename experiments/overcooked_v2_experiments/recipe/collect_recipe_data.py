
import os
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import hydra
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from tqdm import tqdm
from functools import partial

from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from jaxmarl.environments.overcooked_v2.common import Actions, DynamicObject
from overcooked_v2_experiments.ppo.policy import PPOPolicy

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

def get_mask_fn(num_ingredients=3):
    """Return a function that masks recipe info from obs."""
    def mask_fn(obs):
        # obs: (H, W, C)
        # 1. Mask Recipe Layers (Dynamic Goal)
        obs = obs.at[..., -1 - num_ingredients : -1].set(0)
        
        # 2. Mask Static Indicators
        # Heuristic indices for demo_cook_simple (38 channels)
        # 19=Ind, 20=Btn.
        obs = obs.at[..., 19].set(0)
        obs = obs.at[..., 20].set(0)
        
        return obs
    return mask_fn


@hydra.main(version_base=None, config_path="../ppo/config", config_name="base")
def main(config: DictConfig):
    print(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    sp_run_dir = config.get("SP_RUN_DIR", None)
    save_dir = config.get("SAVE_DIR", "/home/myuser/recipe_data")
    layout_name = config.env.ENV_KWARGS.get("layout", "demo_cook_simple")
    
    if not sp_run_dir:
        print("Error: SP_RUN_DIR not specified.")
        return
        
    sp_run_path = Path(sp_run_dir)
    if not sp_run_path.exists():
        print(f"Error: SP_RUN_DIR {sp_run_path} does not exist.")
        return

    # Find checkpoints
    partner_ckpts = []
    run_dirs = sorted([d for d in sp_run_path.iterdir() if d.is_dir() and d.name.startswith("run_")])
    for run_dir in run_dirs:
        ckpt_path = run_dir / "ckpt_final"
        if ckpt_path.exists():
            partner_ckpts.append(str(ckpt_path))
    
    if not partner_ckpts:
        print("No checkpoints found!")
        return

    print(f"Found {len(partner_ckpts)} partners.")
    os.makedirs(save_dir, exist_ok=True)
    
    # --- Setup JAX Environment ---
    env = OvercookedV2(
        layout=layout_name,
        agent_view_size=2,
        random_agent_positions=False,
        sample_recipe_on_delivery=True,
        negative_rewards=True,
    )
    
    # Initialize a dummy policy to get structure
    sample_params, sample_config = load_partner_params(partner_ckpts[0])
    if sample_params is None:
        return
    
    policy_def = PPOPolicy(sample_params, sample_config, stochastic=True)
    
    mask_fn = get_mask_fn(num_ingredients=3)
    
    # 2) Channel Verification (Diagnostic Print)
    print("--- Channel Verification ---")
    dummy_key = jax.random.PRNGKey(0)
    obs0, _ = env.reset(dummy_key)
    ego_obs0 = np.array(obs0["agent_1"])
    print(f"Obs Shape: {ego_obs0.shape}")
    non_zero_channels = np.where(ego_obs0.sum(axis=(0,1)) != 0)[0]
    print(f"Active Channels indices: {non_zero_channels}")
    print("----------------------------")


    # --- JIT Rollout Function ---
    @partial(jax.jit, static_argnums=(2,))
    def run_rollout(rng, partner_params, num_steps=400):
        
        def _env_step(carry, _):
            env_state, last_obs, hstate, _, rng = carry # ignore last_done in carry input
            
            rng, rng_act, rng_step = jax.random.split(rng, 3)
            
            # Partner Action (Agent 0)
            partner_obs = last_obs["agent_0"]
            partner_ac, new_hstate = policy_def.compute_action(
                partner_obs, 
                jnp.bool_(False), # done is passed as False usually in scan unless using auto-reset logic
                hstate, 
                rng_act, 
                params=partner_params
            )
            
            # Ego Action (Agent 1) -> Stay
            # 1) Use Enum
            ego_ac = Actions.stay.value 
            
            actions = {
                "agent_0": partner_ac,
                "agent_1": ego_ac
            }
            
            new_obs, new_state, reward, new_done, info = env.step(rng_step, env_state, actions)
            
            # Data Recording
            raw_ego_obs = last_obs["agent_1"]
            masked_ego_obs = mask_fn(raw_ego_obs)
            
            # 4) Clean Imports: DynamicObject used outside.
            recipe_int = env_state.recipe
            
            # 3) Capture Done
            # We record 'done' to filter segments later.
            done_signal = new_done["__all__"]
            
            step_data = (
                masked_ego_obs, 
                partner_ac, 
                recipe_int,
                done_signal
            )
            
            carry = (new_state, new_obs, new_hstate, done_signal, rng)
            return carry, step_data

        rng, rng_reset = jax.random.split(rng)
        obs, state = env.reset(rng_reset)
        hstate = policy_def.init_hstate(batch_size=1)
        
        dones = {a: jnp.bool_(False) for a in env.agents}
        dones["__all__"] = jnp.bool_(False)
        
        init_carry = (state, obs, hstate, jnp.bool_(False), rng)
        
        final_carry, trajectory = jax.lax.scan(_env_step, init_carry, None, length=num_steps)
        
        return trajectory
    
    
    # --- Data Collection Loop ---
    
    N_EPISODES = 50
    MAX_STEPS = 400
    SEGMENT_K = 10
    NUM_ACTIONS = len(Actions)
    
    run_rollout_vmap = jax.vmap(run_rollout, in_axes=(0, None, None))
    
    total_samples = 0
    
    pbar = tqdm(partner_ckpts, desc="Partners")
    for pid, ckpt_path in enumerate(pbar):
        params, _ = load_partner_params(ckpt_path)
        if params is None:
            continue
            
        master_key = jax.random.PRNGKey(pid * 1000)
        batch_keys = jax.random.split(master_key, N_EPISODES)
        
        # Output: (N_EPISODES, MAX_STEPS, ...)
        # traj_obs: (N, T, H, W, C)
        # traj_act: (N, T)
        # traj_rec: (N, T)
        # traj_done:(N, T)
        traj_obs, traj_act, traj_rec, traj_done = run_rollout_vmap(batch_keys, params, MAX_STEPS)
        
        traj_obs = np.array(traj_obs)
        traj_act = np.array(traj_act)
        traj_rec = np.array(traj_rec)
        traj_done = np.array(traj_done)
        
        batch_obs_segments = []
        batch_act_segments = []
        batch_rec_segments = []
        batch_ptr_segments = []
        
        for i in range(N_EPISODES):
            obs_seq = traj_obs[i]
            act_seq = traj_act[i]
            rec_seq = traj_rec[i]
            done_seq = traj_done[i]
            
            curr_start = 0
            while curr_start + SEGMENT_K <= MAX_STEPS:
                segment_recipes = rec_seq[curr_start : curr_start + SEGMENT_K]
                segment_dones = done_seq[curr_start : curr_start + SEGMENT_K]
                
                # Check 1: Done in segment?
                # If any done=True in the segment, it means it crosses episode boundary or includes termination.
                # Strictly typically we want continuous valid steps.
                # If segment_dones[:-1].any(): done happened BEFORE data ended.
                # If segment_dones[-1] is True: it's the last step. That's fine.
                # But if done happens at index 0..K-2, then next steps are "post-done" (if env doesn't reset).
                # OvercookedV2 step doesn't reset. So post-done states are valid but static/broken?
                # Safest: Skip segment if any done is True.
                if np.any(segment_dones): 
                     # If we want to be very strict: valid segment must NOT contain terminal state, 
                     # OR terminal state is only allowed at K-1?
                     # Let's simple skip if any done occurs to avoid complexity.
                     curr_start += SEGMENT_K
                     continue

                # Check 2: Recipe constant?
                if not np.all(segment_recipes == segment_recipes[0]):
                    curr_start += 1
                    continue
                
                # Check 3: Valid Recipe?
                rec_val = segment_recipes[0]
                rec_idx_list = DynamicObject.get_ingredient_idx_list_jit(rec_val)
                valid_ingredients = [x for x in rec_idx_list if x >= 0]
                if not valid_ingredients:
                    curr_start += 1
                    continue
                
                final_rec_id = valid_ingredients[0]
                
                seg_obs = obs_seq[curr_start : curr_start + SEGMENT_K]
                seg_act = act_seq[curr_start : curr_start + SEGMENT_K]
                
                seg_act_onehot = np.eye(NUM_ACTIONS)[seg_act]
                
                batch_obs_segments.append(seg_obs)
                batch_act_segments.append(seg_act_onehot)
                batch_rec_segments.append(final_rec_id)
                batch_ptr_segments.append(pid)
                
                curr_start += SEGMENT_K
        
        if batch_obs_segments:
            out_file = os.path.join(save_dir, f"partner_{pid}.npz")
            np.savez(
                out_file,
                obs=np.array(batch_obs_segments),
                act=np.array(batch_act_segments),
                recipe=np.array(batch_rec_segments),
                partner=np.array(batch_ptr_segments)
            )
            total_samples += len(batch_obs_segments)
            
    print(f"Data collection complete. Total segments: {total_samples}")

if __name__ == "__main__":
    main()
