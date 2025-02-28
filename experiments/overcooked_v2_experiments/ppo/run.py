import copy
from functools import partial
from pathlib import Path
from omegaconf import OmegaConf
import wandb
import jax
import os
from datetime import datetime
import jax.numpy as jnp

from overcooked_v2_experiments.human_rl.imitation.bc_policy import BCPolicy
from overcooked_v2_experiments.ppo.policy import PPOParams
from overcooked_v2_experiments.ppo.utils.fcp import FCPWrapperPolicy
from .ippo import make_train
from .utils.store import (
    load_all_checkpoints,
    store_checkpoint,
)
from .utils.utils import get_num_devices, get_run_base_dir
from overcooked_v2_experiments.utils.utils import (
    mini_batch_pmap,
    scanned_mini_batch_map,
)
from overcooked_v2_experiments.ppo.utils.visualize_ppo import visualize_ppo_policy

jax.config.update("jax_debug_nans", True)


def load_fcp_populations(population_dir):
    def _load_fcp_population(dir):
        all_checkpoints, fcp_config = load_all_checkpoints(
            dir, final_only=False, skip_initial=True
        )
        all_population_params, _ = jax.tree_util.tree_flatten(
            all_checkpoints, is_leaf=lambda x: type(x) is PPOParams
        )
        print(
            f"Loaded FCP population params for {len(all_population_params)} policies from {dir}"
        )
        all_population_params = jax.tree_util.tree_map(
            lambda *v: jnp.stack(v), *all_population_params
        )
        return all_population_params, fcp_config

    all_populations = []
    first_fcp_config = None
    for dir in population_dir.iterdir():
        if not dir.is_dir() or "fcp_" not in dir.name:
            continue

        print(f"Loading FCP population from {dir}")
        population, fcp_config = _load_fcp_population(dir)
        all_populations.append(population)
        if first_fcp_config is None:
            first_fcp_config = fcp_config

    print(f"Successfully loaded {len(all_populations)} FCP populations")
    all_populations = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *all_populations)
    return all_populations, first_fcp_config


def single_run(config):
    num_seeds = config["NUM_SEEDS"]
    num_runs = num_seeds

    all_populations = None
    if "FCP" in config:
        print("Training FCP")
        assert num_seeds == 1
        print("Loading population from", config["FCP"])
        population_dir = Path(config["FCP"])

        all_populations, fcp_population_config = load_fcp_populations(population_dir)
        all_populations = all_populations.params

        num_runs = jax.tree_util.tree_flatten(all_populations)[0][0].shape[0]

        print(f"Loaded FCP population with {num_runs} runs")

        fcp_params_shape = jax.tree_util.tree_map(lambda x: x.shape, all_populations)
        print("FCP params shape", fcp_params_shape)

    bc_policy = None
    if "BC" in config:
        print("Training with BC")
        layout_name = config["env"]["ENV_KWARGS"]["layout"]
        split = "all"
        run_id = 1
        print(f"Loading BC policy from {layout_name}-{split}-{run_id}")
        bc_policy = BCPolicy.from_pretrained(layout_name, split, run_id)

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_runs)

        config_copy = copy.deepcopy(config)
        if bc_policy is not None:
            config_copy["env"]["ENV_KWARGS"]["force_path_planning"] = True

        population_config = None
        if all_populations is not None:
            population_config = fcp_population_config

        train_func = make_train(
            config_copy,
            population_config=population_config,
        )

        # num_devices = len(jax.devices("gpu"))
        # num_devices = 1
        num_devices = get_num_devices()
        print("Using", num_devices, "devices")

        train_jit = jax.jit(train_func)

        train_extra_args = {}
        if all_populations is not None:
            print("Training with FCP")
            train_extra_args["population"] = all_populations
        elif bc_policy is not None:
            print("Training with BC")
            print("Using BC policy", bc_policy)
            train_extra_args["population"] = bc_policy

        out = mini_batch_pmap(train_jit, num_devices)(rngs, **train_extra_args)
        # out = scanned_mini_batch_map(train_jit, 4, use_pmap=True)(
        #     rngs, **train_extra_args
        # )
        # out = scanned_mini_batch_map(train_jit, 2, num_devices=num_devices)(
        #     rngs, **train_extra_args
        # )
        # out = jax.vmap(train_jit)(rngs, **train_extra_args)

        return out
