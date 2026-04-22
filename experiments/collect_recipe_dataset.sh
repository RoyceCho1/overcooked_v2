#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./experiments/collect_recipe_dataset.sh <SP_PATH> [SAVE_DIR]
#
# SP_PATH supports:
#   - /.../demo_cook_simple_SP
#   - /.../demo_cook_simple_SP/run_0
#   - /.../demo_cook_simple_SP/run_0/ckpt_final

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <SP_PATH> [SAVE_DIR]"
  exit 1
fi

INPUT_PATH="$1"
SAVE_DIR="${2:-}"

if [[ ! -e "$INPUT_PATH" ]]; then
  echo "Error: path does not exist: $INPUT_PATH"
  exit 1
fi

ABS_PATH="$(realpath "$INPUT_PATH")"
BASE_NAME="$(basename "$ABS_PATH")"

# Normalize to SP root directory that contains run_*/ckpt_final
if [[ "$BASE_NAME" == "ckpt_final" ]]; then
  SP_RUN_DIR="$(dirname "$(dirname "$ABS_PATH")")"
elif [[ "$BASE_NAME" =~ ^run_[0-9]+$ ]]; then
  SP_RUN_DIR="$(dirname "$ABS_PATH")"
else
  SP_RUN_DIR="$ABS_PATH"
fi

if [[ -z "$SAVE_DIR" ]]; then
  SAVE_DIR="$(dirname "$SP_RUN_DIR")/recipe_data_demo_cook_simple_obs_only"
fi

# Quick validation
RUN_COUNT="$(find "$SP_RUN_DIR" -maxdepth 1 -type d -name 'run_*' | wc -l | tr -d ' ')"
CKPT_COUNT="$(find "$SP_RUN_DIR" -maxdepth 2 -type d -path '*/run_*/ckpt_final' | wc -l | tr -d ' ')"

if [[ "$RUN_COUNT" == "0" || "$CKPT_COUNT" == "0" ]]; then
  echo "Error: no run_*/ckpt_final found under: $SP_RUN_DIR"
  exit 1
fi

echo "[collect_recipe_dataset] SP_RUN_DIR: $SP_RUN_DIR"
echo "[collect_recipe_dataset] SAVE_DIR:   $SAVE_DIR"
echo "[collect_recipe_dataset] run dirs:   $RUN_COUNT"
echo "[collect_recipe_dataset] ckpt dirs:  $CKPT_COUNT"

action_include="${RECIPE_INCLUDE_PARTNER_ACTION:-false}"
action_visibility="${RECIPE_PARTNER_ACTION_VISIBILITY_AWARE:-false}"
segment_k="${RECIPE_SEGMENT_K:-10}"
segment_stride="${RECIPE_SEGMENT_STRIDE:-10}"
max_steps="${RECIPE_MAX_STEPS:-400}"
episodes="${RECIPE_EPISODES_PER_PARTNER:-50}"

python -m overcooked_v2_experiments.recipe.collect_recipe_data \
  env=demo_cook_simple \
  +SP_RUN_DIR="$SP_RUN_DIR" \
  +SAVE_DIR="$SAVE_DIR" \
  +RECIPE_SEGMENT_K="$segment_k" \
  +RECIPE_SEGMENT_STRIDE="$segment_stride" \
  +RECIPE_MAX_STEPS="$max_steps" \
  +RECIPE_EPISODES_PER_PARTNER="$episodes" \
  +RECIPE_INCLUDE_PARTNER_ACTION="$action_include" \
  +RECIPE_PARTNER_ACTION_VISIBILITY_AWARE="$action_visibility"
