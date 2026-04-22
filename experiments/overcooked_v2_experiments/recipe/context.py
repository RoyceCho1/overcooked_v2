from typing import NamedTuple

import jax
import jax.numpy as jnp


class ContextState(NamedTuple):
    # (num_envs, K, H, W, C)
    obs_buffer: jnp.ndarray
    # (num_envs, K, A)
    act_buffer: jnp.ndarray
    # (num_envs,)
    current_step: jnp.ndarray
    # (num_envs,)
    current_recipe: jnp.ndarray
    # (num_envs, num_classes)
    recipe_ctx: jnp.ndarray
    # (num_envs,)
    valid_mask: jnp.ndarray


class RecipeContextManager:
    """Online recipe-context tracker from ego observations (optional partner actions)."""

    def __init__(
        self,
        encoder_apply_fn,
        encoder_params,
        K=10,
        num_envs=1,
        num_actions=0,
        num_classes=2,
        obs_shape=(5, 5, 26),
        mask_fn=None,
        use_actions=False,
    ):
        self.encoder_apply_fn = encoder_apply_fn
        self.encoder_params = encoder_params
        self.K = K
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.num_classes = num_classes
        self.obs_shape = obs_shape
        self.mask_fn = mask_fn
        self.use_actions = use_actions

        self.default_ctx = jnp.full((self.num_classes,), 1.0 / self.num_classes)

    def init_state(self, current_recipes):
        return ContextState(
            obs_buffer=jnp.zeros((self.num_envs, self.K, *self.obs_shape), dtype=jnp.float32),
            act_buffer=jnp.zeros((self.num_envs, self.K, self.num_actions), dtype=jnp.float32),
            current_step=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            current_recipe=current_recipes,
            recipe_ctx=jnp.tile(self.default_ctx, (self.num_envs, 1)),
            valid_mask=jnp.zeros((self.num_envs,), dtype=jnp.bool_),
        )

    def update(self, state: ContextState, ego_obs, partner_act, current_recipes, dones):
        # 1) Input masking to prevent recipe leakage.
        if self.mask_fn is not None:
            ego_obs = self.mask_fn(ego_obs)

        # 2) Reset buffers when episode ends or recipe changes.
        recipe_changed = current_recipes != state.current_recipe
        should_reset = dones | recipe_changed

        # 3) Normalize action input when enabled.
        if self.use_actions:
            if partner_act.ndim == 1:
                partner_act = jax.nn.one_hot(partner_act, self.num_actions)
        else:
            partner_act = jnp.zeros(
                (self.num_envs, self.num_actions),
                dtype=state.act_buffer.dtype,
            )

        def _write_obs(args):
            buf, step, val = args
            return buf.at[step].set(val)

        def _no_write_obs(args):
            buf, _, _ = args
            return buf

        def _write_act(args):
            buf, step, val = args
            return buf.at[step].set(val)

        def _no_write_act(args):
            buf, _, _ = args
            return buf

        def _update_single(
            s_obs,
            s_act,
            s_step,
            s_ctx,
            s_valid,
            _s_recipe,
            in_obs,
            in_act,
            in_recipe,
            in_reset,
        ):
            curr_step = jnp.where(in_reset, 0, s_step)
            curr_ctx = jnp.where(in_reset, self.default_ctx, s_ctx)
            curr_valid = jnp.where(in_reset, False, s_valid)
            curr_recipe = in_recipe

            do_collect = curr_step < self.K

            new_obs_buf = jax.lax.cond(
                do_collect,
                _write_obs,
                _no_write_obs,
                (s_obs, curr_step, in_obs),
            )

            if self.use_actions:
                new_act_buf = jax.lax.cond(
                    do_collect,
                    _write_act,
                    _no_write_act,
                    (s_act, curr_step, in_act),
                )
            else:
                new_act_buf = s_act

            final_step = curr_step + do_collect
            just_finished = (final_step == self.K) & do_collect

            return (
                new_obs_buf,
                new_act_buf,
                final_step,
                curr_ctx,
                curr_valid,
                curr_recipe,
                just_finished,
            )

        (
            new_obs_buf,
            new_act_buf,
            new_step,
            temp_ctx,
            temp_valid,
            new_recipe,
            just_finished,
        ) = jax.vmap(_update_single)(
            state.obs_buffer,
            state.act_buffer,
            state.current_step,
            state.recipe_ctx,
            state.valid_mask,
            state.current_recipe,
            ego_obs,
            partner_act,
            current_recipes,
            should_reset,
        )

        # 4) Run encoder on all env buffers, then update context for envs that just reached K.
        if self.use_actions:
            logits = self.encoder_apply_fn(
                {"params": self.encoder_params},
                new_obs_buf,
                new_act_buf,
            )
        else:
            logits = self.encoder_apply_fn(
                {"params": self.encoder_params},
                new_obs_buf,
            )

        probs = jax.nn.softmax(logits, axis=-1)

        final_ctx = jnp.where(just_finished[:, None], probs, temp_ctx)
        final_valid = jnp.where(just_finished, True, temp_valid)

        return ContextState(
            obs_buffer=new_obs_buf,
            act_buffer=new_act_buf,
            current_step=new_step,
            current_recipe=new_recipe,
            recipe_ctx=final_ctx,
            valid_mask=final_valid,
        )
