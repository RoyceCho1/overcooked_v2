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
        if not all_population_params:
            raise ValueError(
                f"No checkpoints found in {dir}. "
                f"Expected structure: {dir}/run_X/ckpt_Y. "
                "Please ensure you copied the 'run_X' folder, not just its contents."
            )
        all_population_params = jax.tree_util.tree_map(
            lambda *v: jnp.stack(v), *all_population_params
        )
        return all_population_params, fcp_config

    all_populations = []
    first_fcp_config = None

    # Check if population_dir itself is a population (contains run_X)
    has_runs = any(d.is_dir() and "run_" in d.name for d in population_dir.iterdir())
    if has_runs:
        print(f"Loading single FCP population from {population_dir}")
        population, fcp_config = _load_fcp_population(population_dir)
        all_populations.append(population)
        first_fcp_config = fcp_config
    else:
        # Iterate subdirectories looking for populations
        for dir in population_dir.iterdir():
            if not dir.is_dir() or ("fcp_" not in dir.name and "run_" not in dir.name):
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
        # assert num_seeds == 1 # Allow multiple seeds
        print("Loading population from", config["FCP"])
        population_dir = Path(config["FCP"])

        all_populations, fcp_population_config = load_fcp_populations(population_dir)
        all_populations = all_populations.params

        # Flatten structure (Runs, Ckpts) -> (TotalAgents) for correct indexing
        print("Flattening population (Runs, Ckpts) -> (TotalAgents)")
        all_populations = jax.tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]), 
            all_populations
        )
        
        pop_size = jax.tree_util.tree_flatten(all_populations)[0][0].shape[0]
        print(f"Loaded FCP population with {pop_size} policies")

        fcp_params_shape = jax.tree_util.tree_map(lambda x: x.shape, all_populations)
        print("FCP params shape", fcp_params_shape)
        
        # Broadcast population for each seed
        # Need (num_runs, pop_size, ...)
        print(f"Broadcasting population to {num_runs} seeds/runs")
        all_populations = jax.tree_util.tree_map(
            lambda x: jnp.tile(x[None, ...], (num_runs,) + (1,) * x.ndim),
            all_populations
        )

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

        # Run sequentially (scan) if num_runs > num_devices to avoid OOM
        # This executes 'num_runs' chunks (1 seed per chunk) sequentially
        if num_runs > num_devices:
            print(f"Running {num_runs} seeds sequentially on {num_devices} devices to decrease VRAM usage.")
            out = scanned_mini_batch_map(train_jit, num_runs, num_devices=num_devices)(rngs, **train_extra_args)
        else:
            out = mini_batch_pmap(train_jit, num_devices)(rngs, **train_extra_args)

        return out
