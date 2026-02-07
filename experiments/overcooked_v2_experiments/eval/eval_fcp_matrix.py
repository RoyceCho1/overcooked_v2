import jax
import jax.numpy as jnp
import numpy as np
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import os
import sys
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union

# Add project root to path
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from jaxmarl.environments.overcooked_v2.common import Actions, DynamicObject
import jaxmarl
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper
from overcooked_v2_experiments.ppo.policy import PPOPolicy
from overcooked_v2_experiments.ppo.utils.store import load_checkpoint
from overcooked_v2_experiments.recipe.context import RecipeContextManager
from overcooked_v2_experiments.recipe.masking import get_mask_fn
from overcooked_v2_experiments.ppo.ippo import RecipeEncoder

def decode_recipe(recipe_bits):
    # Pure JAX version to avoid CPU sync
    idxs = DynamicObject.get_ingredient_idx_list_jit(recipe_bits)
    # idxs is likely already a JAX array if it comes from jaxmarl
    # but get_ingredient_idx_list_jit implies JIT compat.
    
    # Logic: return idxs[0] if valid else -1
    # Check if first index is valid (>=0)
    # Assuming -1 is invalid in idxs
    
    first_idx = idxs[0]
    is_valid = first_idx >= 0
    
    return jax.lax.select(is_valid, first_idx, -1)

def make_oracle_context(env_state_recipe):
    # env_state_recipe: scalar int32 (since we are inside eval loop, JAX array of shape (1,) or scalar)
    # DynamicObject.get_ingredient_idx_list_jit is vmapped if input is batched.
    # Here input is from env.step -> likely (1,) or scalar.
    
    # We define a scalar function
    def _get_ctx(recipe):
        idxs = DynamicObject.get_ingredient_idx_list_jit(recipe)
        first_idx = idxs[0] 
        first_idx = jnp.clip(first_idx, 0, 1) 
        return jax.nn.one_hot(first_idx, 2)
        
    # If batched from JAX Env, map it?
    # Evaluation env usually returns shape (Num_envs,). For eval Num_envs=1.
    # So recipe is (1,).
    # We can just apply to [0] or use vmap.
    
    # FIX: Handle scalar input (shape ()) which vmap hates
    if env_state_recipe.ndim == 0:
        env_state_recipe = env_state_recipe[jnp.newaxis]
        
    return jax.vmap(_get_ctx)(env_state_recipe)


# --- Custom FCP Policy Wrapper ---
class FCPPolicy(PPOPolicy):
     def compute_action(self, obs, done, hstate, key, context=None):
        # Flatten batch dim if needed, but here we assume single env evaluation (or batch=1)
        # obs: (B, ...). context: (B, 2)
        
        # If context is provided, append it to inputs for the network
        if context is not None:
             # Network expects (obs, done, ctx)
             ac_in = (obs, done, context)
        else:
             ac_in = (obs, done)

        # Ensure correct dimensionality for network call
        # PPOPolicy logic handles adding batch dims if missing
        # But here we construct the tuple yourself, so we might need to be careful.
        # Let's rely on PPOPolicy.network.apply expecting the tuple.
        
        # We need to manually match the dimension expansion logic of PPOPolicy.compute_action
        # if we bypass it. But PPOPolicy.compute_action typically takes (obs, done, hstate).
        # We can OVERRIDE it or call network directly.
        
        # Let's adapt PPOPolicy logic:
        def _add_dim(tree):
            return jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], tree)

        if not self.with_batching:
             # Add batch dim (1, ...)
             ac_in = _add_dim(ac_in)
             hstate_in = hstate[jnp.newaxis, ...]
        else:
             hstate_in = hstate
        
        # Call network
        next_hstate, pi, _ = self.network.apply(self.params, hstate_in, ac_in)

        if self.stochastic:
            action = pi.sample(seed=key)
        else:
            action = jnp.argmax(pi.probs, axis=-1)

        if not self.with_batching:
            action = action[0]
            next_hstate = next_hstate[0]

        return action, next_hstate


def load_policy(run_dir, run_id, checkpoint="final") -> FCPPolicy:
    dir_path = Path(run_dir)
    # load_checkpoint handles appending run_{id}
    config, params = load_checkpoint(dir_path, run_id, checkpoint)
    print(f"DEBUG: Policy loaded from {dir_path.name}, FCP={config.get('FCP')}, context_mode={config.get('context_mode')}")
    # Option 1: Set with_batching=True so we handle dims manually in run_episode
    return FCPPolicy(params, config=config, stochastic=True, with_batching=True) # Eval usually stochastic? Or deterministic? User requested "Avg return", usually mean of stochastic runs.


# --- Custom Rollout Function ---
def run_episode(
    env, 
    fcp_agent: FCPPolicy, 
    sp_agent: PPOPolicy, 
    key, 
    context_mode: str = "encoder", # "encoder", "oracle", "none"
    context_manager: Optional[RecipeContextManager] = None,
    encoder_params=None
):
    """
    Simulates one episode between FCP agent (idx 0 or 1? Usually FCP is Agent 1 for eval?)
    Prompt says: "agent1 is always FCP policy, agent0 is always SP partner"
    """
    obs, state = env.reset(key)
    hstate_fcp = fcp_agent.init_hstate(1) # Batch 1
    hstate_sp = sp_agent.init_hstate(1)
    
    done = False
    total_reward = 0
    timestep = 0
    
    # --- JIT-Compiled Step Function for Speed ---
    @jax.jit
    def step_fn(state, obs, hstate_sp, hstate_fcp, ctx_state, key, last_done):
        # 1. Prepare Keys
        key, k1, k2 = jax.random.split(key, 3)
        
        # 2. Prepare Observations (Fixed for JIT)
        obs0_arr = obs["agent_0"]
        obs1_arr = obs["agent_1"]
        
        # Add dimensions for RNN: (1, 1, H, W, C)
        # Assuming obs input is already array or we ensure it outside
        # But step_fn input obs is from env.step output which is JAX friendly
        obs0_in = obs0_arr[jnp.newaxis, jnp.newaxis, ...]
        obs1_in = obs1_arr[jnp.newaxis, jnp.newaxis, ...]
        
        # 3. SP Agent Action
        action0, next_hstate_sp = sp_agent.compute_action(
            obs0_in, last_done, hstate_sp, k1
        )
        
        # 4. FCP Agent Context & Action
        ctx_input = None
        
        if context_mode == "encoder":
            if ctx_state is not None:
                ctx_input = ctx_state.recipe_ctx # (Batch, Feat) -> (1, 2)
                # Add Time dimension at index 1 -> (Batch, Time, Feat)
                ctx_input = ctx_input[:, jnp.newaxis, :]  # (1, 1, 2)
        elif context_mode == "oracle":
             # Generate Oracle context from state
             # state.env_state.recipe -> (1,)
             batch_ctx = make_oracle_context(state.env_state.recipe) # (1, 2)
             ctx_input = batch_ctx[:, jnp.newaxis, :] # (1, 1, 2)
        else:
             # None
             ctx_input = None
            
        action1, next_hstate_fcp = fcp_agent.compute_action(
            obs1_in, last_done, hstate_fcp, k2, context=ctx_input
        )
        
        # 5. Env Step
        # Squeeze Actions to Scalar/0-d
        a0_scalar = action0[0, 0]
        a1_scalar = action1[0, 0]
        actions = {"agent_0": a0_scalar, "agent_1": a1_scalar}
        
        next_obs, next_state, reward, done, info = env.step(key, state, actions)
        
        # 6. Context Update (If Needed)
        next_ctx_state = ctx_state
        if context_mode == "encoder" and context_manager is not None:
             act_dim = 6
             # One-hot encode action0 (Partner)
             a0_onehot = jax.nn.one_hot(a0_scalar, act_dim)
             act_in = a0_onehot[jnp.newaxis, ...] # (1, 6)
             
             obs_in = obs0_arr[jnp.newaxis, ...] # (1, ...)
             
             current_rec_id = decode_recipe(next_state.env_state.recipe)
             rec_in = jnp.array([current_rec_id])
             
             done_val_t = done["__all__"]
             done_arr_in = jnp.array([done_val_t])
             
             next_ctx_state = context_manager.update(
                 ctx_state,
                 obs_in,
                 act_in, 
                 rec_in, 
                 done_arr_in 
             )
             
        # Process rewards
        # PPO returns rewards as dict usually? Or Array?
        if isinstance(reward, dict):
             r1 = reward["agent_1"]
        else:
             r1 = reward[1] if reward.shape[0]>1 else reward[0]
             
        done_val = done["__all__"] if isinstance(done, dict) else done
        next_last_done = jnp.array([[done_val]]) # (1, 1)

        return (next_state, next_obs, next_hstate_sp, next_hstate_fcp, next_ctx_state, key, next_last_done, r1, done_val, actions)


    # Initialize Context
    ctx_state = None
    if context_mode == "encoder" and context_manager is not None:
        init_recipe = jnp.array([decode_recipe(state.env_state.recipe)])
        ctx_state = context_manager.init_state(init_recipe)

    done_flag = False
    last_done = jnp.array([[False]]) 
    
    key, subkey = jax.random.split(key)
    # print("DEBUG: Starting Episode Loop (JIT Accelerated)")
    
    timestep = 0
    # Loop - Still Python loop but calling JIT function
    while not done_flag and timestep < 405: # Safety break
        # Call JIT Step
        (state, obs, hstate_sp, hstate_fcp, ctx_state, key, last_done, r_t, done_bool, debug_acts) = step_fn(
            state, obs, hstate_sp, hstate_fcp, ctx_state, key, last_done
        )
        
        total_reward += float(r_t)
        done_flag = bool(done_bool)
        
        timestep += 1
        
        # if timestep % 100 == 0:
        #      pass # Can print if needed but avoiding I/O is faster
             
    return total_reward


@hydra.main(version_base=None, config_path="../ppo/config", config_name="base")
def main(config):
    # Setup Paths
    fcp_enc_dir = "/home/myuser/runs/FCP_SP_population_env96"
    fcp_base_dir = "/home/myuser/runs/FCP_SP_population_no_encoder"
    fcp_oracle_dir = "/home/myuser/runs/FCP_oracle_demo_cook_simple" # NEW
    sp_partner_dir = "/home/myuser/runs/SP_population_eval_38ch_fixed_v2"
    
    # Override from config if provided (optional)
    if config.get("FCP_ENC_DIR"): fcp_enc_dir = config.FCP_ENC_DIR
    if config.get("FCP_BASE_DIR"): fcp_base_dir = config.FCP_BASE_DIR
    if config.get("FCP_ORACLE_DIR"): fcp_oracle_dir = config.FCP_ORACLE_DIR
    if config.get("SP_PARTNER_DIR"): sp_partner_dir = config.SP_PARTNER_DIR
    
    layout_name = config.env.ENV_KWARGS.layout # "demo_cook_simple"
    
    print(f"--- FCP Comparison Matrix Evaluation ---")
    print(f"Layout: {layout_name}")
    # print(f"FCP(+Enc): {fcp_enc_dir}")
    # print(f"FCP(Base): {fcp_base_dir}")
    print(f"FCP(Oracle): {fcp_oracle_dir}")
    print(f"SP Partners: {sp_partner_dir}")
    
    # Load Policies
    # Assuming run_0 to run_7
    seeds = range(8)
    
    # fcp_enc_policies = [load_policy(fcp_enc_dir, s) for s in seeds]
    # fcp_base_policies = [load_policy(fcp_base_dir, s) for s in seeds]
    fcp_oracle_policies = [load_policy(fcp_oracle_dir, s) for s in seeds]
    sp_policies = [load_policy(sp_partner_dir, s) for s in seeds]
    
    print("Policies Loaded.")

    # Setup Environment
    # FIX 2: Use jaxmarl.make + Wrapper to match training env exactly
    # FIX 4: Explicitly override agent_view_size in a clean dict
    env_kwargs = dict(config.env.ENV_KWARGS)
    env_kwargs["agent_view_size"] = 2
    
    print(f"DEBUG: Initializing Env with kwargs: {env_kwargs}")
    env = jaxmarl.make(config.env.ENV_NAME, **env_kwargs)
    env = OvercookedV2LogWrapper(env, replace_info=False)
    
    # DEBUG SHAPE
    obs_shape = env.observation_space().shape
    print(f"DEBUG: EVAL ENV Obs Shape: {obs_shape}")  # EXPECT (5, 5, 38)
    if obs_shape[0] != 5 or obs_shape[1] != 5:
        print("CRITICAL WARNING: Env Obs Shape is NOT 5x5! Agent will likely fail.")
    
    # Setup Context Manager (Only for FCP+Enc) - Disabled for Oracle Eval
    context_manager = None 
    encoder_params = None

    # --- Matrix 3: FCP(Oracle) vs SP ---
    R_oracle = np.zeros((8, 8))
    print("\nEvaluating FCP(Oracle) vs SP...")
    for i in seeds: 
        for j in seeds:
            print(f"--- FCP(Oracle) Seed {i} vs SP Partner {j} ---")
            rng = jax.random.PRNGKey(i*100 + j + 8000)
            rewards = []
            for ep in tqdm(range(20), desc=f"Ep (FCP_oracle_{i}_vs_SP_{j})"):
                rng, subkey = jax.random.split(rng)
                r = run_episode(
                    env, 
                    fcp_oracle_policies[i], 
                    sp_policies[j], 
                    subkey, 
                    context_mode="oracle", # Use Oracle context
                    context_manager=None, # Not needed
                    encoder_params=None
                )
                rewards.append(r)
            avg_r = np.mean(rewards)
            R_oracle[i, j] = avg_r
            print(f"FCP_oracle_{i} vs SP_{j}: {avg_r:.2f}")

    # --- Metrics & Output ---
    print("\n\n=== Final ZSC Evaluation Results (Oracle Only) ===")
    
    # Partner-wise means
    oracle_sp_means = np.mean(R_oracle, axis=0)
    
    # Construct DataFrame
    df = pd.DataFrame({
        "Partner": [f"SP_{j}" for j in seeds],
        "Avg Return FCP(Oracle)": oracle_sp_means,
    })
    
    print("\n[Partner-wise Performance Table]")
    print(df.to_string(index=False, float_format="%.2f"))
    
    # Global Means
    oracle_global = np.mean(R_oracle)
    
    print(f"\n[Global Performance]")
    print(f"FCP(Oracle) Global Avg: {oracle_global:.2f}")

    # Save to file
    save_path = Path("eval_results_fcp_oracle_matrix.csv")
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()
