from functools import partial
from overcooked_v2_experiments.eval.policy import AbstractPolicy
from overcooked_v2_experiments.ppo.policy import PPOParams, PPOPolicy
import chex
import jax.numpy as jnp
import jax


@chex.dataclass
class HStateWrapper:
    hstates: chex.Array
    params_idxs: chex.Scalar


class FCPWrapperPolicy(AbstractPolicy):
    population: PPOParams
    population_size: chex.Scalar

    policy: PPOPolicy

    def __init__(self, config, *params: PPOParams):
        stacked_params = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *params)
        population_size = jax.tree_util.tree_leaves(stacked_params)[0].shape[0]

        policy = PPOPolicy(params=None, config=config)

        self.population = stacked_params
        self.population_size = population_size
        self.policy = policy

    def compute_action(self, obs, done, hstate, key):
        params_idxs, hstates = hstate.params_idxs, hstate.hstates

        batch_size = hstates.shape[0]

        key, subkey = jax.random.split(key)
        new_params_idxs = self._sample_param_idxs(batch_size, subkey)
        params_idxs = jnp.where(done, new_params_idxs, params_idxs)

        def _compute_action(policy_idx, obs, done, hstate, key):
            params = self._get_params(policy_idx)
            print("test:", obs.shape, done.shape, hstate.shape, key.shape, type(params))
            return self.policy.compute_action(obs, done, hstate, key, params=params)

        action_keys = jax.random.split(key, batch_size)
        actions, next_hstates = jax.vmap(_compute_action)(
            params_idxs, obs, done, hstates, action_keys
        )

        return actions, HStateWrapper(hstates=next_hstates, params_idxs=params_idxs)

    def init_hstate(self, batch_size, key):
        assert key is not None

        params_idxs = self._sample_param_idxs(batch_size, key)

        hstate_init_func = partial(self.policy.init_hstate, batch_size=1)
        policy_hstates = jax.vmap(hstate_init_func, axis_size=batch_size)()

        hstate = HStateWrapper(hstates=policy_hstates, params_idxs=params_idxs)
        return hstate

    def _sample_param_idxs(self, batch_size, key):
        params_idxs = jax.random.randint(key, (batch_size,), 0, self.population_size)
        return params_idxs

    def _get_params(self, policy_idx):
        params = jax.tree_util.tree_map(lambda x: x[policy_idx], self.population)
        return params.params
