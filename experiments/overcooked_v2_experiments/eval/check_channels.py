
import jax
import jax.numpy as jnp
import numpy as np
import hydra
import os
import sys
from pathlib import Path

# Add project root to path
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from jaxmarl.environments.overcooked_v2.common import DynamicObject
from overcooked_v2_experiments.ppo.utils.store import load_checkpoint
from overcooked_v2_experiments.recipe.context import RecipeContextManager
from overcooked_v2_experiments.recipe.masking import get_mask_fn
from overcooked_v2_experiments.ppo.ippo import RecipeEncoder
from overcooked_v2_experiments.eval.eval_fcp_matrix import FCPPolicy, load_policy, decode_recipe

@hydra.main(version_base=None, config_path="../ppo/config", config_name="base")
def main(config):
    # Paths (Defaulting to the ones we know)
    fcp_enc_dir = "/home/myuser/runs/FCP_SP_population_env96"
    sp_partner_dir = "/home/myuser/runs/SP_population_eval_38ch_fixed_v2"
    
    # Allow override
    if config.get("FCP_ENC_DIR"): fcp_enc_dir = config.FCP_ENC_DIR
    if config.get("SP_PARTNER_DIR"): sp_partner_dir = config.SP_PARTNER_DIR
    
    try:
        layout_name = config.env.ENV_KWARGS.layout
    except Exception:
        layout_name = "demo_cook_simple"
        print("Warning: No layout specified in config. Defaulting to 'demo_cook_simple'.")
    
    print(f"=== Channel Inspection Tool ===")
    print(f"Layout: {layout_name}")
    print(f"Loading FCP Agent from: {fcp_enc_dir}")
    print(f"Loading SP Agent from: {sp_partner_dir}")

    # Load 1 seed each
    fcp_agent = load_policy(fcp_enc_dir, 0)
    sp_agent = load_policy(sp_partner_dir, 0)
    print("Agents Loaded Successfully.")
    
    print(f"\n[SP Agent Config]")
    try:
        # PPOPolicy stores config in self.config usually? No, PPOPolicy(actor_critic, params, config)
        import pprint
        pprint.pprint(sp_agent.config)
        # Check specific Env Kwargs
        if "env" in sp_agent.config:
            print("Env Config:", sp_agent.config["env"])
    except:
        print("Could not print config.")

    # Init Env
    env = OvercookedV2(layout=layout_name)
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    
    print(f"\n[Environment Raw Output]")
    print(f"Agent 0 (SP) Raw Obs Shape: {obs['agent_0'].shape}")
    print(f"Agent 1 (FCP) Raw Obs Shape: {obs['agent_1'].shape}")
    
    # Context Logic
    init_recipe = jnp.array([decode_recipe(state.recipe)])
    
    # Dummy Encoder for Context Manager
    dummy_encoder = RecipeEncoder()
    # Use dummy params since we just want shape check
    obs_shape = env.observation_space().shape
    dummy_obs = jnp.zeros((1, 10, *obs_shape)) 
    dummy_act = jnp.zeros((1, 10, 6))
    dummy_params = dummy_encoder.init(jax.random.PRNGKey(0), dummy_obs, dummy_act)['params']
    
    context_manager = RecipeContextManager(
        encoder_apply_fn=dummy_encoder.apply,
        encoder_params=dummy_params,
        K=10, num_envs=1, num_actions=6,
        obs_shape=obs_shape, mask_fn=get_mask_fn(num_ingredients=3)
    )
    ctx_state = context_manager.init_state(init_recipe)
    
    print(f"\n[Context Manager]")
    print(f"Recipe Context Vector Shape: {ctx_state.recipe_ctx.shape}")
    
    # Simulating Input Construction (Eval Matrix Logic)
    print(f"\n[Constructing Inputs]")
    
    # --- SP Agent Input ---
    obs0_arr = jnp.array(obs["agent_0"])
    # SP is standard PPO, expects Raw Obs (38ch)
    print(f"-> SP Agent Input (Obs0): {obs0_arr.shape} (Channels: {obs0_arr.shape[-1]})")
    
    # Check SP Agent Param Shape (First Conv Layer)
    try:
        # PPOPolicy stores params. 'params' is a FrozenDict.
        # Structure varies, usually params['params']['CNN_0']['Conv_0']['kernel']
        # Let's try to inspect safely
        import flax
        flat_params = flax.traverse_util.flatten_dict(sp_agent.params, sep='/')
        conv_kernel_key = [k for k in flat_params.keys() if 'Conv_0/kernel' in k][0]
        conv_kernel = flat_params[conv_kernel_key]
        print(f"-> SP Agent Network expects Input Channels: {conv_kernel.shape[-2]} (Kernel Shape: {conv_kernel.shape})")
    except Exception as e:
        print(f"Could not inspect SP Agent params directly: {e}")

    # --- FCP Agent Input ---
    obs1_arr = jnp.array(obs["agent_1"])
    ctx_input = ctx_state.recipe_ctx # (1, 2)
    
    print(f"-> FCP Agent Raw Obs Input: {obs1_arr.shape}")
    print(f"-> FCP Agent Context Input: {ctx_input.shape}")
    print(f"   (FCP Agent Policy internally handles obs + context concatenation)")
    
    # Check FCP Agent Param Shape
    try:
        flat_params_fcp = flax.traverse_util.flatten_dict(fcp_agent.params, sep='/')
        conv_kernel_key_fcp = [k for k in flat_params_fcp.keys() if 'Conv_0/kernel' in k][0]
        conv_kernel_fcp = flat_params_fcp[conv_kernel_key_fcp]
        print(f"-> FCP Agent Network expects Input Channels: {conv_kernel_fcp.shape[-2]} (Kernel Shape: {conv_kernel_fcp.shape})")
    except Exception as e:
        print(f"Could not inspect FCP Agent params directly: {e}")

    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    main()
