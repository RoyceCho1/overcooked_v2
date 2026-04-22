from pathlib import Path
import hydra
import sys
import os
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
import wandb

from overcooked_v2_experiments.ppo.policy import PPOParams
from overcooked_v2_experiments.ppo.utils.store import load_all_checkpoints, store_checkpoint
from overcooked_v2_experiments.ppo.run import single_run
from overcooked_v2_experiments.ppo.tune import tune
from overcooked_v2_experiments.ppo.utils.utils import get_run_base_dir
from overcooked_v2_experiments.ppo.utils.visualize_ppo import visualize_ppo_policy

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))


jax.config.update("jax_debug_nans", True)


def single_run_with_viz(config):
    config = OmegaConf.to_container(config)
    num_checkpoints = config["NUM_CHECKPOINTS"]
    model_name = config["model"]["TYPE"]
    layout_name = config["env"]["ENV_KWARGS"]["layout"]
    agent_view_size = config["env"]["ENV_KWARGS"].get("agent_view_size", None)
    avs_str = f"avs-{agent_view_size}" if agent_view_size is not None else "avs-full"
    run_name = f"ippo_{model_name}_ov2_{layout_name}_{avs_str}"
    if "FCP" in config:
        population_dir = Path(config["FCP"])
        run_name = f"FCP_{population_dir.name}"


    with wandb.init(
        entity=config["wandb"]["ENTITY"],
        project=config["wandb"]["PROJECT"],
        tags=["IPPO", model_name, "OvercookedV2"],
        config=config,
        mode=config["wandb"]["WANDB_MODE"],
        name=run_name,
    ) as run:
        run_id = run.id
        run_base_dir = get_run_base_dir(run_id, config)
        config["RUN_BASE_DIR"] = run_base_dir

        out = single_run(config)

    if config["NUM_CHECKPOINTS"] > 0:
        checkpoints = out["runner_state"][1]
        num_runs = jax.tree_util.tree_flatten(checkpoints)[0][0].shape[0]
        for run_num in range(num_runs):
            for checkpoint in range(num_checkpoints):
                params = jax.tree_util.tree_map(
                    lambda x: x[run_num][checkpoint], checkpoints
                )

                store_checkpoint(
                    config,
                    params,
                    run_num,
                    checkpoint,
                    final=checkpoint == num_checkpoints - 1,
                )

    if config["VISUALIZE"]:
        visualize_ppo_policy(
            run_base_dir,
            key=jax.random.PRNGKey(config["SEED"]),
            final_only=True,
            num_seeds=2,
        )

        visualize_ppo_policy(
            run_base_dir,
            key=jax.random.PRNGKey(config["SEED"]),
            final_only=True,
            num_seeds=500,
            cross=True,
            no_viz=True,
        )

@hydra.main(version_base=None, config_path="config", config_name="base")
def main(config):
    print(config)

    if config["TUNE"]:
        tune(config)
    elif "NUM_ITERATIONS" in config:
        # Import lazily so regular SP/OP/FCP runs do not require BC/human data deps.
        from overcooked_v2_experiments.ppo.state_sample_run import state_sample_run

        state_sample_run(config)
    else:
        single_run_with_viz(config)


if __name__ == "__main__":
    main()
