# 평가 엔진 & 시각화 진입점. 정책 페어링을 받아 환경을 돌려 점수·프레임 시퀀스를 생성.
import argparse
from collections import defaultdict
from typing import List
import jaxmarl
import sys
import os
import itertools
import jax.numpy as jnp
import jax
import copy
from datetime import datetime
from pathlib import Path
import chex
import imageio
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
import csv

from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from .policy import AbstractPolicy, PolicyPairing
from .rollout import get_rollout
from .rollout import get_rollout
from .utils import get_recipe_identifier


@chex.dataclass
class PolicyVizualization:
    frame_seq: chex.Array
    total_reward: chex.Scalar


def visualize_pairing(
    output_dir: Path,
    policies: PolicyPairing,
    layout_name,
    key,
    env_kwargs={},
    num_seeds=1,
    all_recipes=False,
    no_viz=False,
    no_csv=False,
):
    runs = eval_pairing(
        policies, layout_name, key, env_kwargs, num_seeds, all_recipes, no_viz
    )

    reward_sum = 0.0
    rows = []
    for annotation, viz in runs.items():
        frame_seq = viz.frame_seq
        total_reward = viz.total_reward

        if not no_viz:
            viz_dir = output_dir / "visualizations"
            os.makedirs(viz_dir, exist_ok=True)
            viz_filename = viz_dir / f"{annotation}.gif"

            imageio.mimsave(viz_filename, frame_seq, "GIF", duration=0.5)

        reward_sum += total_reward
        rows.append([annotation, total_reward])
        print(f"\t{annotation}:\t{total_reward}")
    reward_mean = reward_sum / len(runs)
    print(f"\tMean reward:\t{reward_mean}")

    if not no_csv:
        summery_name = "reward_summary.csv"
        summery_file = output_dir / summery_name
        with open(summery_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            fieldnames = ["annotation", "total_reward"]

            writer.writerow(fieldnames)

            for row in rows:
                writer.writerow(row)
        print(f"Summary written to {summery_file}")


def eval_pairing(
    policies: PolicyPairing,
    layout_name,
    key,
    env_kwargs={},
    num_seeds=1,
    all_recipes=False,
    no_viz=False,
):
    assert (
        not all_recipes and num_seeds > 1
    ), "Only one of all_recipes and num_seeds can be set"
    assert "layout" not in env_kwargs, "Layout should be passed as layout_name"

    if all_recipes:
        layout = overcooked_v2_layouts[layout_name]
        env_kwargs.pop("layout")

        possible_recipes = jnp.array(layout.possible_recipes)

        def _rollout_recipe(recipe):
            _layout = copy.deepcopy(layout)
            _layout.possible_recipes = [recipe]
            env = OvercookedV2(layout=_layout, **env_kwargs)

            rollout = get_rollout(policies, env, key)

            return rollout

        rollouts = jax.vmap(_rollout_recipe)(possible_recipes)
        annotations = [
            "recipe-" + get_recipe_identifier(r) for r in layout.possible_recipes
        ]

    else:
        env = OvercookedV2(layout=layout_name, **env_kwargs)

        def _rollout_seed(key):
            return get_rollout(policies, env, key)

        keys = jax.random.split(key, num_seeds)
        rollouts = jax.vmap(_rollout_seed)(keys)
        annotations = [f"seed-{i}" for i in range(num_seeds)]

    if no_viz:
        frame_seqs = [None] * len(annotations)
    else:
        agent_view_size = env_kwargs.get("agent_view_size", None)
        viz = OvercookedV2Visualizer()
        frame_seqs = jax.vmap(viz.render_sequence, in_axes=(0, None))(
            rollouts.state_seq, agent_view_size
        )

    return {
        annotation: PolicyVizualization(
            frame_seq=frame_seqs[i], total_reward=rollouts.total_reward[i]
        )
        for i, annotation in enumerate(annotations)
    }
