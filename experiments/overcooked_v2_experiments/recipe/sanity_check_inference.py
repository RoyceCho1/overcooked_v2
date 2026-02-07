
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import os
from pathlib import Path

from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from jaxmarl.environments.overcooked_v2.common import Actions, DynamicObject
from overcooked_v2_experiments.ppo.policy import PPOPolicy
from overcooked_v2_experiments.recipe.recipe_encoder_jax import RecipeEncoder
from overcooked_v2_experiments.recipe.context import RecipeContextManager
from overcooked_v2_experiments.recipe.masking import get_mask_fn

def load_partner_policy(path):
    ckpt_dir = Path(path)
    if ckpt_dir.name == "checkpoint":
        ckpt_dir = ckpt_dir.parent
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    ckpt = orbax_checkpointer.restore(ckpt_dir, item=None)
    policy = PPOPolicy(ckpt["params"], ckpt.get("config", {}), stochastic=True)
    return policy

def decode_recipe(recipe_bits):
    idxs = DynamicObject.get_ingredient_idx_list_jit(recipe_bits)
    idxs = np.array(idxs)
    return int(idxs[0]) if len(idxs[idxs>=0]) > 0 else -1

def main():
    print("--- Sanity Check: Recipe Context Inference ---")
    
    # 1. Setup Env and Partner
    env = OvercookedV2(layout="demo_cook_simple", agent_view_size=2, sample_recipe_on_delivery=True)
    
    # Find a partner
    sp_dir = Path("/home/myuser/runs/SP_population_20251205")
    run_dir = next(sp_dir.glob("run_0")) # Use first partner
    ckpt_path = run_dir / "ckpt_final"
    print(f"Loading Partner from {ckpt_path}")
    partner_policy = load_partner_policy(ckpt_path)
    
    # 2. Setup Encoder and Context Manager
    encoder_path = "/home/myuser/recipe_encoder_ckpt_v2"
    print(f"Loading Encoder from {encoder_path}")
    
    dummy_encoder = RecipeEncoder()
    # Init dummy params to get structure
    dummy_obs = jnp.zeros((1, 10, 5, 5, 38))
    dummy_act = jnp.zeros((1, 10, 6))
    dummy_params = dummy_encoder.init(jax.random.PRNGKey(0), dummy_obs, dummy_act)['params']
    
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    encoder_params = orbax_checkpointer.restore(encoder_path, item=dummy_params)
    
    mask_fn = get_mask_fn(num_ingredients=3)
    
    ctx_manager = RecipeContextManager(
        encoder_apply_fn=dummy_encoder.apply,
        encoder_params=encoder_params,
        K=10,
        num_envs=1,
        num_actions=6,
        obs_shape=(5, 5, 38),
        mask_fn=mask_fn
    )
    
    # 3. Rollout Loop
    print("\nStarting Rollout...")
    rng = jax.random.PRNGKey(42)
    obs_dict, state = env.reset(rng)
    
    # Init Context
    init_recipe = jnp.array([decode_recipe(state.recipe)])
    ctx_state = ctx_manager.init_state(init_recipe)
    
    hstate = partner_policy.init_hstate(batch_size=1)
    
    true_recipe_history = []
    pred_recipe_history = []
    
    for step in range(100):
        # Partner acts
        partner_obs = obs_dict["agent_0"]
        rng, subkey = jax.random.split(rng)
        
        # Partner policy expects unbatched or batched? 
        # Usually PPOPolicy handles it. Let's pass array.
        partner_obs_jnp = jnp.array(partner_obs)
        partner_act, hstate = partner_policy.compute_action(partner_obs_jnp, jnp.bool_(False), hstate, subkey)
        partner_act_int = int(partner_act)
        
        # Ego acts (Stay)
        ego_act = Actions.stay.value
        
        # Step Env
        actions = {"agent_0": partner_act_int, "agent_1": ego_act}
        rng, step_key = jax.random.split(rng)
        obs_dict, state, _, dones, _ = env.step(step_key, state, actions)
        
        # Update Context
        # Need batched inputs for manager update (num_envs=1)
        # partner_obs: (1, H, W, C)
        # partner_act: (1,)
        # current_rec: (1,)
        # done: (1,)
        
        batch_p_obs = jnp.array(partner_obs_jnp)[None, ...]
        batch_p_act = jnp.array([partner_act_int])
        
        current_recipe_id = decode_recipe(state.recipe)
        batch_rec = jnp.array([current_recipe_id])
        batch_done = jnp.array([dones["__all__"]])
        
        ctx_state = ctx_manager.update(
            ctx_state, 
            batch_p_obs, 
            batch_p_act, 
            batch_rec, 
            batch_done
        )
        
        # Log
        true_rec = current_recipe_id
        ctx_probs = ctx_state.recipe_ctx[0] # (2,)
        pred_rec = np.argmax(ctx_probs)
        confidence = ctx_probs[pred_rec]
        
        true_recipe_history.append(true_rec)
        pred_recipe_history.append(pred_rec)
        
        print(f"Step {step:03d} | True: {true_rec} | Pred: {pred_rec} (Conf: {confidence:.2f}) | Ctx: [{ctx_probs[0]:.2f}, {ctx_probs[1]:.2f}]")
        
        if dones["__all__"]:
            print("Episode Done. Resetting...")
            rng, r_key = jax.random.split(rng)
            obs_dict, state = env.reset(r_key)
            init_recipe = jnp.array([decode_recipe(state.recipe)])
            # Manually reset ctx state or rely on update logic?
            # update logic handles done=True in NEXT update.
            # But we just passed done=True. So ctx state SHOULD have triggered reset internally for next step buffers.
            # But ctx vector itself might reset to prior on done?
            # Let's see behavior.
    
    # Summary
    print("\n--- Summary ---")
    # Check accuracy after step 10
    total = 0
    correct = 0
    for i in range(10, len(true_recipe_history)):
        total += 1
        if true_recipe_history[i] == pred_recipe_history[i]:
            correct += 1
            
    print(f"Accuracy (Step 10+): {correct}/{total} ({correct/total*100:.1f}%)")

if __name__ == "__main__":
    main()
