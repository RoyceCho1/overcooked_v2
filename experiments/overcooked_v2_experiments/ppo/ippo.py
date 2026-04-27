""" 
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Optional, Sequence, NamedTuple, Any, Dict, Union
from flax.training.train_state import TrainState
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper, OvercookedV2LogWrapper
import hydra
from omegaconf import OmegaConf
from datetime import datetime
import os
import wandb
import functools
import math
import pickle
from overcooked_v2_experiments.ppo.models.rnn import ScannedRNN
import matplotlib.pyplot as plt
from jaxmarl.environments.overcooked_v2.overcooked import ObservationType
from overcooked_v2_experiments.eval.policy import AbstractPolicy
from overcooked_v2_experiments.ppo.models.abstract import ActorCriticBase
from .models.model import get_actor_critic, initialize_carry
from overcooked_v2_experiments.eval.policy import AbstractPolicy
from flax import core


# ... (Imports)
from overcooked_v2_experiments.recipe.recipe_encoder_jax import RecipeEncoder
from overcooked_v2_experiments.recipe.context import RecipeContextManager
import orbax.checkpoint as ocp
from jaxmarl.environments.overcooked_v2.common import DynamicObject, Actions
from overcooked_v2_experiments.recipe.masking import get_mask_fn

def build_recipe_codebook(env):
    possible_recipes = jnp.array(env.layout.possible_recipes, dtype=jnp.int32)
    recipe_codes = jax.vmap(DynamicObject.get_recipe_encoding)(possible_recipes)
    return recipe_codes


def recipe_bits_to_label(recipe_bits, recipe_codes):
    matches = recipe_codes == recipe_bits
    has_match = jnp.any(matches)
    label = jnp.argmax(matches.astype(jnp.int32)).astype(jnp.int32)
    return jnp.where(has_match, label, jnp.int32(0))


def decode_recipe_batch(recipe_bits_batch, recipe_codes):
    return jax.vmap(lambda bits: recipe_bits_to_label(bits, recipe_codes))(recipe_bits_batch)


def make_oracle_context(env_state_recipe, recipe_codes):
    recipe_ids = decode_recipe_batch(env_state_recipe, recipe_codes)
    return jax.nn.one_hot(recipe_ids, recipe_codes.shape[0])


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    train_mask: jnp.ndarray
    recipe_ctx: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(
    config,
    update_step_offset=None,
    update_step_num_overwrite=None,
    population_config=None,
):
    env_config = config["env"]
    model_config = config["model"]

    env = jaxmarl.make(env_config["ENV_NAME"], **env_config["ENV_KWARGS"])

    model_config["NUM_ACTORS"] = env.num_agents * model_config["NUM_ENVS"]
    model_config["NUM_UPDATES"] = (
        model_config["TOTAL_TIMESTEPS"]
        // model_config["NUM_STEPS"]
        // model_config["NUM_ENVS"]
    )
    model_config["MINIBATCH_SIZE"] = (
        model_config["NUM_ACTORS"]
        * model_config["NUM_STEPS"]
        // model_config["NUM_MINIBATCHES"]
    )

    num_checkpoints = config["NUM_CHECKPOINTS"]
    checkpoint_steps = jnp.linspace(
        0,
        model_config["NUM_UPDATES"],
        num_checkpoints,
        endpoint=True,
        dtype=jnp.int32,
    )
    if num_checkpoints > 0:
        # make sure the last checkpoint is the last update step
        checkpoint_steps = checkpoint_steps.at[-1].set(model_config["NUM_UPDATES"])

    print("Checkpoint steps: ", checkpoint_steps)

    def _update_checkpoint(checkpoint_states, params, i):
        jax.debug.print("Saving checkpointing {i}", i=i)
        return jax.tree_util.tree_map(
            lambda x, y: x.at[i].set(y),
            checkpoint_states,
            params,
        )

    # Context Setup
    context_mode = str(config.get("context_mode", "encoder")).lower()
    use_recipe_encoder = bool(
        config.get("USE_RECIPE_ENCODER", config.get("use_recipe_encoder", True))
    )
    encoder_use_actions = bool(
        config.get(
            "RECIPE_ENCODER_USE_ACTIONS",
            config.get("recipe_encoder_use_actions", False),
        )
    )
    encoder_action_dim = int(config.get("RECIPE_ENCODER_ACTION_DIM", len(Actions)))
    encoder_k = int(config.get("RECIPE_ENCODER_K", 10))
    encoder_path = config.get(
        "RECIPE_ENCODER_PATH",
        "/home/myuser/recipe_encoder_ckpt_v2",
    )

    use_fcp = bool("FCP" in config and config["FCP"])
    use_context = bool(use_fcp and context_mode in ("encoder", "oracle"))

    if context_mode == "oracle":
        use_recipe_encoder = False
    if context_mode == "none":
        use_recipe_encoder = False

    recipe_codes = build_recipe_codebook(env)
    num_recipe_classes = int(recipe_codes.shape[0])
    obs_shape = env.observation_space().shape

    # Recipe Encoder Setup
    model_action_dim = encoder_action_dim if encoder_use_actions else 0
    dummy_encoder = RecipeEncoder(
        num_actions=model_action_dim,
        num_classes=num_recipe_classes,
        use_actions=encoder_use_actions,
    )
    dummy_obs = jnp.zeros((1, encoder_k, *obs_shape))
    if encoder_use_actions:
        dummy_act = jnp.zeros((1, encoder_k, model_action_dim))
        dummy_params = dummy_encoder.init(jax.random.PRNGKey(0), dummy_obs, dummy_act)[
            "params"
        ]
    else:
        dummy_params = dummy_encoder.init(jax.random.PRNGKey(0), dummy_obs)["params"]

    encoder_params = dummy_params
    if use_context and context_mode == "encoder" and use_recipe_encoder:
        print(f"Loading Recipe Encoder from {encoder_path}")
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        if os.path.exists(encoder_path):
            encoder_params = orbax_checkpointer.restore(encoder_path, item=dummy_params)
        else:
            print(f"Warning: Encoder path {encoder_path} not found. Using dummy params.")
    else:
        print(
            "Recipe Encoder disabled. "
            f"(context_mode={context_mode}, use_context={use_context}, "
            f"use_recipe_encoder={use_recipe_encoder})"
        )

    context_manager = None
    if use_context and context_mode == "encoder":
        context_manager = RecipeContextManager(
            encoder_apply_fn=dummy_encoder.apply,
            encoder_params=encoder_params,
            K=encoder_k,
            num_envs=model_config["NUM_ENVS"],
            num_actions=model_action_dim,
            num_classes=num_recipe_classes,
            obs_shape=obs_shape,
            mask_fn=get_mask_fn(num_ingredients=env.layout.num_ingredients),
            use_actions=encoder_use_actions,
        )
    else:
        print(f"Recipe Context Manager: DISABLED ({context_mode} mode)")

    env = OvercookedV2LogWrapper(env, replace_info=False)

    def create_learning_rate_fn():
        base_learning_rate = model_config["LR"]

        lr_warmup = model_config["LR_WARMUP"]
        update_steps = model_config["NUM_UPDATES"]
        warmup_steps = int(lr_warmup * update_steps)

        steps_per_epoch = (
            model_config["NUM_MINIBATCHES"] * model_config["UPDATE_EPOCHS"]
        )

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=warmup_steps * steps_per_epoch,
        )
        cosine_epochs = max(update_steps - warmup_steps, 1)

        print("Update steps: ", update_steps)
        print("Warmup epochs: ", warmup_steps)
        print("Cosine epochs: ", cosine_epochs)

        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
        )
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps * steps_per_epoch],
        )
        return schedule_fn

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0,
        end_value=0.0,
        transition_steps=model_config["REW_SHAPING_HORIZON"],
    )

    # If FCP is provided, we are training an Ego agent against a population.
    # The user requested to fix the Ego agent to index 1 (Right).
    if "FCP" in config and config["FCP"]:
        train_idxs = jnp.ones((model_config["NUM_ENVS"],), dtype=jnp.int32) * 1
    else:
        # For Self-Play, we want to train on both positions (distributed across envs).
        train_idxs = jnp.linspace(
            0,
            env.num_agents,
            model_config["NUM_ENVS"],
            dtype=jnp.int32,
            endpoint=False,
        )
    train_mask_dict = {a: train_idxs == i for i, a in enumerate(env.agents)}
    train_mask_flat = batchify(
        train_mask_dict, env.agents, model_config["NUM_ACTORS"]
    ).squeeze()

    print("train_mask_flat", train_mask_flat.shape)
    print("train_mask_flat sum", train_mask_flat.sum())

    use_population_annealing = False
    if "POPULATION_ANNEAL_HORIZON" in config:
        print("Using population annealing")
        use_population_annealing = True
        transition_begin = 0
        if "POPULATION_ANNEAL_BEGIN" in config:
            transition_begin = config["POPULATION_ANNEAL_BEGIN"]

        anneal_horizon = config["POPULATION_ANNEAL_HORIZON"]
        if anneal_horizon == 0:
            population_annealing_schedule = optax.constant_schedule(1.0)
        else:
            population_annealing_schedule = optax.linear_schedule(
                init_value=0.0,
                end_value=1.0,
                transition_steps=config["POPULATION_ANNEAL_HORIZON"] - transition_begin,
                transition_begin=transition_begin,
            )

    def train(
        rng,
        population: Optional[Union[AbstractPolicy, core.FrozenDict[str, Any]]] = None,
        initial_train_state=None,
    ):
        original_seed = rng[0]

        jax.debug.print("original_seed {s}", s=rng)

        # INIT NETWORK
        network = get_actor_critic(config)

        rng, _rng = jax.random.split(rng)

        if use_context:
            init_x = (
                jnp.zeros(
                    (1, model_config["NUM_ENVS"], *env.observation_space().shape),
                ),
                jnp.zeros((1, model_config["NUM_ENVS"])),
                jnp.zeros((1, model_config["NUM_ENVS"], num_recipe_classes)),
            )
        else:
            init_x = (
                jnp.zeros(
                    (1, model_config["NUM_ENVS"], *env.observation_space().shape),
                ),
                jnp.zeros((1, model_config["NUM_ENVS"])),
            )
        init_hstate = initialize_carry(config, model_config["NUM_ENVS"])

        if init_hstate is not None:
            print("init_hstate", init_hstate.shape)
        # jax.debug.print("check1 {x}", x=init_hstate.flatten()[0])

        print("init_x", init_x[0].shape, init_x[1].shape)

        network_params = network.init(_rng, init_hstate, init_x)
        if model_config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(model_config["MAX_GRAD_NORM"]),
                optax.adam(create_learning_rate_fn(), eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(model_config["MAX_GRAD_NORM"]),
                optax.adam(model_config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        if initial_train_state is not None:
            train_state = initial_train_state

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, model_config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        
        # --- Context Init ---
        current_recipes = decode_recipe_batch(env_state.env_state.recipe, recipe_codes)
        if context_manager is not None:
            context_state = context_manager.init_state(current_recipes)
        else:
            context_state = None
        # --------------------

        init_hstate = initialize_carry(config, model_config["NUM_ACTORS"])
        # jax.debug.print("check2 {x}", x=init_hstate.flatten()[0])

        init_population_hstate = None
        init_population_annealing_mask = None
        if population is not None:
            is_policy_population = False
            if isinstance(population, AbstractPolicy):
                is_policy_population = True
                rng, _rng = jax.random.split(rng)
                init_population_hstate = population.init_hstate(
                    model_config["NUM_ACTORS"], key=_rng
                )
            else:
                assert (
                    population_config is not None
                ), "population_config cannot be None if population is not a policy"
                population_network = get_actor_critic(population_config)
                init_population_hstate = initialize_carry(
                    population_config, model_config["NUM_ACTORS"]
                )

                fcp_population_size = jax.tree_util.tree_flatten(population)[0][
                    0
                ].shape[0]
                print("FCP population size", fcp_population_size)

                # print(f"normal hstate {init_hstate.shape}")
                # print(f"population hstate {init_population_hstate.shape}")

            if use_population_annealing:

                def _sample_population_annealing_mask(step, rng):
                    return jax.random.uniform(
                        rng, (model_config["NUM_ENVS"],)
                    ) < population_annealing_schedule(step)

                def _make_train_mask(annealing_mask):
                    full_anneal_mask = jnp.tile(annealing_mask, env.num_agents)
                    return jnp.where(full_anneal_mask, train_mask_flat, True)

                rng, _rng = jax.random.split(rng)
                init_population_annealing_mask = _sample_population_annealing_mask(
                    0, _rng
                )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            (
                train_state,
                checkpoint_states,
                env_state,
                last_obs,
                last_done,
                update_step,
                initial_hstate,
                initial_population_hstate,
                last_population_annealing_mask,
                initial_fcp_pop_agent_idxs,
                rng,
                context_state, # NEW
            ) = runner_state

            # jax.debug.print("check3 {x}", x=initial_hstate.flatten()[0])

            # COLLECT TRAJECTORIES
            def _env_step(env_step_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    update_step,
                    hstate,
                    population_hstate,
                    population_annealing_mask,
                    fcp_pop_agent_idxs,
                    rng,
                    context_state, # NEW
                ) = env_step_state


                # jax.debug.print("check4 {x}", x=hstate.flatten()[0])

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *env.observation_space().shape
                )

                # --- Context ---
                if context_mode == "encoder" and context_state is not None:
                    recipe_ctx_batch = jnp.tile(context_state.recipe_ctx, (env.num_agents, 1))
                elif context_mode == "oracle" and use_context:
                    batch_ctx = make_oracle_context(
                        env_state.env_state.recipe,
                        recipe_codes,
                    )
                    recipe_ctx_batch = jnp.tile(batch_ctx, (env.num_agents, 1))
                else:
                    recipe_ctx_batch = jnp.zeros(
                        (model_config["NUM_ACTORS"], num_recipe_classes)
                    )
                # ---------------

                if use_context:
                    ac_in = (
                        obs_batch[np.newaxis, :],
                        last_done[np.newaxis, :],
                        recipe_ctx_batch[np.newaxis, :],
                    )
                else:
                    ac_in = (
                        obs_batch[np.newaxis, :],
                        last_done[np.newaxis, :],
                    )

                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)

                # jax.debug.print("check5 {x}", x=hstate.flatten()[0])

                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                action_pick_mask = jnp.ones(
                    (model_config["NUM_ACTORS"],), dtype=jnp.bool_
                )
                if population is not None:
                    print("Using population")

                    obs_population = obs_batch
                    if isinstance(population, AbstractPolicy):
                        obs_featurized = jax.vmap(
                            env.get_obs_for_type, in_axes=(0, None)
                        )(env_state.env_state, ObservationType.FEATURIZED)
                        obs_population = batchify(
                            obs_featurized, env.agents, model_config["NUM_ACTORS"]
                        )

                    if is_policy_population:
                        rng, _rng = jax.random.split(rng)
                        pop_actions, population_hstate = population.compute_action(
                            obs_population, last_done, population_hstate, _rng
                        )
                    else:

                        def _compute_population_actions(
                            policy_idx, obs_pop, obs_ld, fcp_h_state
                        ):
                            current_p = jax.tree.map(
                                lambda x: x[policy_idx], population
                            )
                            current_ac_in = (
                                obs_pop[np.newaxis, np.newaxis, :],
                                jnp.array([obs_ld])[np.newaxis, :],
                            )
                            new_fcp_h_state, fcp_pi, _ = population_network.apply(
                                current_p,
                                jax.tree.map(lambda x: x[np.newaxis, :], fcp_h_state),
                                current_ac_in,
                            )
                            fcp_action = fcp_pi.sample(seed=_rng)
                            return fcp_action.squeeze(), jax.tree.map(
                                lambda x: x.squeeze(axis=0), new_fcp_h_state
                            )

                        pop_actions, population_hstate = jax.vmap(
                            _compute_population_actions
                        )(
                            fcp_pop_agent_idxs,
                            obs_population,
                            last_done,
                            population_hstate,
                        )

                    action_pick_mask = train_mask_flat
                    if use_population_annealing:
                        action_pick_mask = _make_train_mask(population_annealing_mask)

                    # use action_pick_mask to select the action from the population or the network
                    action = jnp.where(action_pick_mask, action, pop_actions)

                env_act = unbatchify(
                    action, env.agents, model_config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # STEP ENV
                recipes_before_step = None
                if context_mode == "encoder" and context_manager is not None:
                    recipes_before_step = decode_recipe_batch(
                        env_state.env_state.recipe,
                        recipe_codes,
                    )

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, model_config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                
                # --- Update Context ---
                ego_obs_for_encoder = last_obs[env.agents[1]]   # agent_1 obs (ego POV)
                partner_act = env_act[env.agents[0]]            # agent_0 action

                if context_mode == "encoder" and context_manager is not None:
                    recipes_after_step = decode_recipe_batch(
                        env_state.env_state.recipe,
                        recipe_codes,
                    )
                    context_state = context_manager.update(
                        context_state,
                        ego_obs_for_encoder,
                        partner_act,
                        recipes_before_step,
                        done["__all__"],
                        next_recipes=recipes_after_step,
                    )
                # ----------------------

                original_reward = jnp.array([reward[a] for a in env.agents])

                current_timestep = (
                    update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
                )
                anneal_factor = rew_shaping_anneal(current_timestep)
                reward = jax.tree_util.tree_map(
                    lambda x, y: x + y * anneal_factor, reward, info["shaped_reward"]
                )

                shaped_reward = jnp.array(
                    [info["shaped_reward"][a] for a in env.agents]
                )
                combined_reward = jnp.array([reward[a] for a in env.agents])

                info["shaped_reward"] = shaped_reward
                info["original_reward"] = original_reward
                info["anneal_factor"] = jnp.full_like(shaped_reward, anneal_factor)
                info["combined_reward"] = combined_reward

                info = jax.tree_util.tree_map(
                    lambda x: x.reshape((model_config["NUM_ACTORS"])), info
                )

                done_batch = batchify(
                    done, env.agents, model_config["NUM_ACTORS"]
                ).squeeze()

                if use_population_annealing:
                    env_steps = (
                        update_step
                        * model_config["NUM_STEPS"]
                        * model_config["NUM_ENVS"]
                    )
                    rng, _rng = jax.random.split(rng)
                    new_population_annealing_mask = jnp.where(
                        done["__all__"],
                        _sample_population_annealing_mask(env_steps, _rng),
                        population_annealing_mask,
                    )
                else:
                    new_population_annealing_mask = population_annealing_mask

                if population is not None and not is_policy_population:
                    new_fcp_pop_agent_idxs = jnp.where(
                        jnp.tile(done["__all__"], env.num_agents),
                        jax.random.randint(
                            _rng, (model_config["NUM_ACTORS"],), 0, fcp_population_size
                        ),
                        fcp_pop_agent_idxs,
                    )
                else:
                    new_fcp_pop_agent_idxs = fcp_pop_agent_idxs

                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, model_config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    action_pick_mask,
                    recipe_ctx_batch, # NEW
                )

                # jax.debug.print("check6 {x}", x=hstate.flatten()[0])

                env_step_state = (
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    update_step,
                    hstate,
                    population_hstate,
                    new_population_annealing_mask,
                    new_fcp_pop_agent_idxs,
                    rng,
                    context_state, # NEW
                )
                return env_step_state, transition

            env_step_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                initial_hstate,
                initial_population_hstate,
                last_population_annealing_mask,
                initial_fcp_pop_agent_idxs,
                rng,
                context_state, # NEW: Add context_state
            )
            env_step_state, traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, model_config["NUM_STEPS"]
            )
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                next_initial_hstate,
                next_population_hstate,
                last_population_annealing_mask,
                next_fcp_pop_agent_idxs,
                rng,
                context_state, # NEW: Unpack updated context_state
            ) = env_step_state

            # jax.debug.print("check7 {x}", x=next_initial_hstate)

            # print("Hilfeeeee", traj_batch.done.shape, traj_batch.action.shape)

            # CALCULATE ADVANTAGE
            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                -1, *env.observation_space().shape
            )
            
            # --- Context for Last Value ---
            # Use the UPDATED context_state from env_step_state
            if context_state is not None:
                recipe_ctx_batch = jnp.tile(context_state.recipe_ctx, (env.num_agents, 1))
            else:
                 recipe_ctx_batch = jnp.zeros((model_config["NUM_ACTORS"], num_recipe_classes))
            # ------------------------------

            if use_context:
                ac_in = (
                    last_obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    recipe_ctx_batch[np.newaxis, :],
                )
            else:
                ac_in = (
                    last_obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )

            _, _, last_val = network.apply(
                train_state.params, next_initial_hstate, ac_in
            )

            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward + model_config["GAMMA"] * next_value * (1 - done) - value
                    )
                    gae = (
                        delta
                        + model_config["GAMMA"]
                        * model_config["GAE_LAMBDA"]
                        * (1 - done)
                        * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        hstate = init_hstate
                        if hstate is not None:
                            hstate = hstate.squeeze(axis=0)
                            # hstate = jax.lax.stop_gradient(hstate)

                        train_mask = True
                        if population is not None:
                            train_mask = jax.lax.stop_gradient(traj_batch.train_mask)

                        # RERUN NETWORK
                        if use_context:
                            ac_in = (traj_batch.obs, traj_batch.done, traj_batch.recipe_ctx)
                        else:
                            ac_in = (traj_batch.obs, traj_batch.done)

                        _, pi, value = network.apply(
                            params,
                            hstate,
                            ac_in,
                        )

                        print("value shape", value.shape)
                        print("targets shape", targets.shape)
                        print("pi shape", pi.logits.shape)

                        log_prob = pi.log_prob(traj_batch.action)

                        # def safe_mean(x, mask):
                        #     x_safe = jnp.where(mask, x, 0.0)
                        #     total = jnp.sum(x_safe)
                        #     count = jnp.sum(mask)
                        #     return total / count

                        # def safe_std(x, mask, eps=1e-8):
                        #     m = safe_mean(x, mask)
                        #     diff_sq = (x - m) ** 2
                        #     variance = safe_mean(diff_sq, mask)
                        #     return jnp.sqrt(variance + eps)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-model_config["CLIP_EPS"], model_config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean(where=train_mask)
                        # value_loss = safe_mean(value_loss, train_mask)

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean(where=train_mask)) / (
                            gae.std(where=train_mask) + 1e-8
                        )
                        # gae_mean = safe_mean(gae, train_mask)
                        # gae_std = safe_std(gae, train_mask)
                        # gae = (gae - gae_mean) / (gae_std + 1e-8)

                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - model_config["CLIP_EPS"],
                                1.0 + model_config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean(where=train_mask)
                        # loss_actor = safe_mean(loss_actor, train_mask)
                        entropy = pi.entropy().mean(where=train_mask)
                        # entropy = safe_mean(pi.entropy(), train_mask)
                        ratio = ratio.mean(where=train_mask)
                        # ratio = safe_mean(ratio, train_mask)

                        total_loss = (
                            loss_actor
                            + model_config["VF_COEF"] * value_loss
                            - model_config["ENT_COEF"] * entropy
                        )

                        return total_loss, (value_loss, loss_actor, entropy, ratio)

                    def _perform_update():
                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params,
                            init_hstate,
                            traj_batch,
                            advantages,
                            targets,
                        )

                        # jax.debug.print(
                        #     "grads {x}, hstate {y}, mask {z}",
                        #     x=jax.tree_util.tree_flatten(grads)[0][0][0],
                        #     y=init_hstate.flatten()[0],
                        #     z=traj_batch.train_mask.sum(),
                        # )

                        new_train_state = train_state.apply_gradients(grads=grads)
                        return new_train_state, total_loss

                    def _no_op():
                        # jax.debug.print("No update")
                        return train_state, (0.0, (0.0, 0.0, 0.0, 0.0))

                    # jax.debug.print(
                    #     "train_mask {x}, {y}",
                    #     x=traj_batch.train_mask.sum(),
                    #     y=traj_batch.train_mask.any(),
                    # )

                    train_state, total_loss = jax.lax.cond(
                        traj_batch.train_mask.any(),
                        _perform_update,
                        _no_op,
                    )
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = (
                    update_state
                )
                rng, _rng = jax.random.split(rng)

                num_actors = model_config["NUM_ACTORS"]

                hstate = init_hstate
                if hstate is not None:
                    print("hstate shape", hstate.shape)
                    hstate = hstate[jnp.newaxis, :]
                    print("hstate shape", hstate.shape)

                batch = (
                    hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                # print(
                #     "batch shapes:",
                #     batch[0].shape,
                #     batch[1].obs.shape,
                #     batch[1].done.shape,
                #     batch[2].shape,
                #     batch[3].shape,
                # )
                # print("hstate shape", hstate.shape)

                permutation = jax.random.permutation(_rng, num_actors)

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], model_config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )

                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            rng, _rng = jax.random.split(rng)
            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                _rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, model_config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info

            total_loss, aux_data = loss_info
            value_loss, loss_actor, entropy, ratio = aux_data

            metric["total_loss"] = total_loss
            metric["value_loss"] = value_loss
            metric["loss_actor"] = loss_actor
            metric["entropy"] = entropy
            metric["ratio"] = ratio

            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)

            update_step += 1
            metric["update_step"] = update_step
            metric["env_step"] = (
                update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
            )

            def callback(metric, original_seed):
                metric.update(
                    {f"rng{int(original_seed)}/{k}": v for k, v in metric.items()}
                )
                wandb.log(metric)
                print(f"[Seed {int(original_seed)}] Update: {metric['update_step']} | Env Step: {metric['env_step']} | Reward: {metric['combined_reward']:.2f}")

            jax.debug.callback(callback, metric, original_seed)

            if num_checkpoints > 0:
                checkpoint_idx_selector = checkpoint_steps == update_step
                checkpoint_states = jax.lax.cond(
                    jnp.any(checkpoint_idx_selector),
                    _update_checkpoint,
                    lambda c, _p, _i: c,
                    checkpoint_states,
                    train_state.params,
                    jnp.argmax(checkpoint_idx_selector),
                )

            runner_state = (
                train_state,
                checkpoint_states,
                env_state,
                last_obs,
                last_done,
                update_step,
                next_initial_hstate,
                next_population_hstate,
                last_population_annealing_mask,
                next_fcp_pop_agent_idxs,
                rng,
                context_state, # NEW
            )
            return runner_state, metric

        initial_update_step = 0
        if update_step_offset is not None:
            initial_update_step = update_step_offset

        initial_checkpoints = jax.tree_util.tree_map(
            lambda p: jnp.zeros((num_checkpoints,) + p.shape, p.dtype),
            train_state.params,
        )

        if num_checkpoints > 0:
            initial_checkpoints = jax.lax.cond(
                (checkpoint_steps[0] == 0) & (initial_update_step == 0),
                _update_checkpoint,
                lambda c, _p, _i: c,
                initial_checkpoints,
                train_state.params,
                0,
            )

        init_fcp_pop_idxs = None
        if population is not None and not is_policy_population:
            init_fcp_pop_idxs = jax.random.randint(
                _rng, (model_config["NUM_ACTORS"],), 0, fcp_population_size
            )

        rng, _rng = jax.random.split(rng)

        runner_state = (
            train_state,
            initial_checkpoints,
            env_state,
            obsv,
            jnp.zeros((model_config["NUM_ACTORS"]), dtype=bool),
            initial_update_step,
            init_hstate,
            init_population_hstate,
            init_population_annealing_mask,
            init_fcp_pop_idxs,
            _rng,
            context_state, # NEW
        )
        num_update_steps = model_config["NUM_UPDATES"]
        if update_step_num_overwrite is not None:
            num_update_steps = update_step_num_overwrite
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, num_update_steps
        )

        # jax.debug.print("Runner state {x}", x=runner_state)
        # jax.debug.print("neg5 {x}", x=runner_state[-5])
        return {"runner_state": runner_state, "metrics": metric}

    return train
