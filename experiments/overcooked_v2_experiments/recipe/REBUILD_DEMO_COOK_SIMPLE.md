# Rebuild Pipeline (demo_cook_simple)

This document describes the end-to-end rerun order on a fresh server.

## 0) Environment setup

```bash
python3 -m pip install -e JaxMARL
python3 -m pip install -e experiments
```

If `hydra`, `jax`, `flax`, or `orbax` import errors appear, install from:

```bash
python3 -m pip install -r experiments/requirements/requirements.txt
```

## 1) Train SP population

```bash
python3 experiments/overcooked_v2_experiments/ppo/main.py \
  +experiment=rnn-sp \
  +env=demo_cook_simple \
  NUM_SEEDS=10
```

Output is under `runs/<wandb_run_id>/run_*`.

## 2) Collect recipe dataset (ego = agent_1, obs-only default)

```bash
python3 experiments/overcooked_v2_experiments/recipe/collect_recipe_data.py \
  +env=demo_cook_simple \
  +SP_RUN_DIR=/abs/path/to/sp_runs \
  +SAVE_DIR=/abs/path/to/recipe_data_demo_cook_simple_obs_only \
  +RECIPE_SEGMENT_K=10 \
  +RECIPE_EPISODES_PER_PARTNER=50 \
  +RECIPE_INCLUDE_PARTNER_ACTION=false
```

Optional action-aware collection:

```bash
python3 experiments/overcooked_v2_experiments/recipe/collect_recipe_data.py \
  +env=demo_cook_simple \
  +SP_RUN_DIR=/abs/path/to/sp_runs \
  +SAVE_DIR=/abs/path/to/recipe_data_demo_cook_simple_obs_act \
  +RECIPE_INCLUDE_PARTNER_ACTION=true \
  +RECIPE_PARTNER_ACTION_VISIBILITY_AWARE=true
```

## 3) Train recipe encoder

Obs-only:

```bash
python3 experiments/overcooked_v2_experiments/recipe/train_recipe_encoder_jax.py \
  --data_dir /abs/path/to/recipe_data_demo_cook_simple_obs_only \
  --save_path /abs/path/to/recipe_encoder_ckpt_demo_cook_simple_obs_only \
  --epochs 50
```

Obs+action:

```bash
python3 experiments/overcooked_v2_experiments/recipe/train_recipe_encoder_jax.py \
  --data_dir /abs/path/to/recipe_data_demo_cook_simple_obs_act \
  --save_path /abs/path/to/recipe_encoder_ckpt_demo_cook_simple_obs_act \
  --epochs 50 \
  --use_actions
```

## 4) Train FCP variants

Base FCP (no context):

```bash
python3 experiments/overcooked_v2_experiments/ppo/main.py \
  +experiment=rnn-fcp-base \
  +env=demo_cook_simple \
  +FCP=/abs/path/to/sp_runs \
  NUM_SEEDS=1
```

Encoder FCP:

```bash
python3 experiments/overcooked_v2_experiments/ppo/main.py \
  +experiment=rnn-fcp \
  +env=demo_cook_simple \
  +FCP=/abs/path/to/sp_runs \
  +RECIPE_ENCODER_PATH=/abs/path/to/recipe_encoder_ckpt_demo_cook_simple_obs_only \
  NUM_SEEDS=1
```

Oracle FCP:

```bash
python3 experiments/overcooked_v2_experiments/ppo/main.py \
  +experiment=rnn-fcp-oracle \
  +env=demo_cook_simple \
  +FCP=/abs/path/to/sp_runs \
  NUM_SEEDS=1
```

## 5) Compare

Compare `FCP_base`, `FCP_encoder`, and `FCP_oracle` with the same evaluation protocol and partner pool.
