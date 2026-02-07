
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import chex
from typing import NamedTuple, Optional

class ContextState(NamedTuple):
    # Buffers for K steps
    # obs_buffer: (num_envs, K, H, W, C)
    obs_buffer: jnp.ndarray
    # act_buffer: (num_envs, K, num_actions)
    act_buffer: jnp.ndarray
    # current_step: (num_envs,) - how many steps collected for current order
    current_step: jnp.ndarray
    # current_recipe: (num_envs,) - current recipe ID to detect changes
    current_recipe: jnp.ndarray
    # recipe_ctx: (num_envs, 2) - current context vector (softmax probs)
    recipe_ctx: jnp.ndarray
    # valid_mask: (num_envs,) - whether context is valid (collected K steps)
    valid_mask: jnp.ndarray

class RecipeContextManager:
    def __init__(self, encoder_apply_fn, encoder_params, K=10, num_envs=1, num_actions=6, obs_shape=(5, 5, 26), mask_fn=None):
        self.encoder_apply_fn = encoder_apply_fn
        self.encoder_params = encoder_params
        self.K = K
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.obs_shape = obs_shape
        self.mask_fn = mask_fn
        
        # Default context is uniform prior
        self.default_ctx = jnp.array([0.5, 0.5])

    def init_state(self, current_recipes):
        """
        Initialize the context state.
        Args:
            current_recipes: (num_envs,) int array of current recipe IDs
        """
        return ContextState(
            obs_buffer=jnp.zeros((self.num_envs, self.K, *self.obs_shape)),
            act_buffer=jnp.zeros((self.num_envs, self.K, self.num_actions)),
            current_step=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            current_recipe=current_recipes,
            recipe_ctx=jnp.tile(self.default_ctx, (self.num_envs, 1)),
            valid_mask=jnp.zeros((self.num_envs,), dtype=jnp.bool_)
        )

    def update(self, state: ContextState, partner_obs, partner_act, current_recipes, dones):
        """
        Update the context state with new observations and actions.
        Args:
            state: Current ContextState
            partner_obs: (num_envs, H, W, C)
            partner_act: (num_envs, num_actions) or (num_envs,) if int
            current_recipes: (num_envs,)
            dones: (num_envs,) boolean
        Returns:
            new_state: Updated ContextState
        """
        # Apply masking if provided (Critical for distribution matching)
        if self.mask_fn is not None:
            partner_obs = self.mask_fn(partner_obs)
            
        
        # 1. Check for resets (done or recipe change)
        # Recipe change: current_recipe != state.current_recipe
        # Done: episode ended
        
        recipe_changed = current_recipes != state.current_recipe
        should_reset = dones | recipe_changed
        
        # If reset, clear buffer and step count, reset context to default
        # BUT: if we already had a valid context, maybe keep it until new one is ready?
        # Requirement: "encoder is rollout-time only... use stored recipe_ctx"
        # Usually, when order changes, we don't know the new recipe yet, so context should probably reset to prior.
        
        # Logic:
        # If should_reset:
        #   current_step = 0
        #   valid_mask = False
        #   recipe_ctx = default (0.5, 0.5)
        #   current_recipe = new_recipe
        
        # 2. Update buffers
        # Only update if not valid yet (step < K)
        # If valid, we stop collecting (or could continue sliding window? Requirement says "initial K steps")
        # "partner's initial K-step trajectory" -> Stop after K.
        
        collecting = state.current_step < self.K
        
        # Update step count
        new_step = jnp.where(should_reset, 0, state.current_step + collecting)
        
        # Update buffers
        # We need to insert at 'current_step' index for each env
        # This is tricky in JAX with batching.
        # We can use .at[...].set() but indices vary per env.
        # Easier to use vmap or scan if possible, but this is inside a step.
        # Let's use vmap for the update logic.
        
        def _update_single(s_obs, s_act, s_step, s_ctx, s_valid, s_rec, 
                           in_obs, in_act, in_rec, in_reset):
            
            # Reset logic
            curr_step = jnp.where(in_reset, 0, s_step)
            curr_ctx = jnp.where(in_reset, self.default_ctx, s_ctx)
            curr_valid = jnp.where(in_reset, False, s_valid)
            curr_rec = in_rec # Always update to current recipe
            
            do_collect = (curr_step < self.K)
            
            # Safe write using lax.cond
            def _write_obs(args):
                buf, step, val = args
                return buf.at[step].set(val)
            
            def _no_write_obs(args):
                buf, _, _ = args
                return buf
                
            new_obs_buf = jax.lax.cond(do_collect, _write_obs, _no_write_obs, (s_obs, curr_step, in_obs))
            
            def _write_act(args):
                buf, step, val = args
                return buf.at[step].set(val)
                
            def _no_write_act(args):
                buf, _, _ = args
                return buf

            new_act_buf = jax.lax.cond(do_collect, _write_act, _no_write_act, (s_act, curr_step, in_act))
            
            # If not collecting (and not reset), keep old buffer. 
            # But wait, if reset happened, s_obs/s_act are old buffers.
            # If reset, we want to start writing to index 0 of OLD buffer (effectively clearing/overwriting).
            # So passing s_obs is correct.
            
            # Increment step
            final_step = curr_step + do_collect
            
            # Check if we just finished K steps
            just_finished = (final_step == self.K) & do_collect
            
            return new_obs_buf, new_act_buf, final_step, curr_ctx, curr_valid, curr_rec, just_finished

        # Vectorize over envs
        # partner_act might need one-hot encoding if it comes as int
        if partner_act.ndim == 1:
            partner_act = jax.nn.one_hot(partner_act, self.num_actions)
            
        new_obs_buf, new_act_buf, new_step, temp_ctx, temp_valid, new_rec, just_finished = jax.vmap(_update_single)(
            state.obs_buffer, state.act_buffer, state.current_step, state.recipe_ctx, state.valid_mask, state.current_recipe,
            partner_obs, partner_act, current_recipes, should_reset
        )
        
        # 3. Run Encoder if needed
        # We need to run encoder for envs where just_finished is True
        # But JAX requires static shapes. So we might have to run on all or mask.
        # Or run on all and only update context where just_finished is True.
        
        # Prepare input for encoder: (N, K, H, W, C)
        # Encoder expects (Batch, K, H, W, C)
        
        # Run encoder on ALL buffers (efficient enough in batch)
        # logits: (N, 2)
        logits = self.encoder_apply_fn({'params': self.encoder_params}, new_obs_buf, new_act_buf)
        probs = jax.nn.softmax(logits, axis=-1)
        
        # Update context only where just_finished is True
        # If just_finished, use new probs. Else keep temp_ctx (which is either old ctx or reset default)
        final_ctx = jnp.where(just_finished[:, None], probs, temp_ctx)
        final_valid = jnp.where(just_finished, True, temp_valid)
        
        return ContextState(
            obs_buffer=new_obs_buf,
            act_buffer=new_act_buf,
            current_step=new_step,
            current_recipe=new_rec,
            recipe_ctx=final_ctx,
            valid_mask=final_valid
        )
