#!/usr/bin/env bash
set -euo pipefail

# Run this inside the Docker container from /home/myuser/overcooked_v2_experiments.
# It collects a fixed-context online-FCP recipe dataset and trains a fresh K=6
# obs-only recipe encoder from that dataset.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
K_ROOT="${K_ROOT:-${WORKDIR}/runs/k6}"

SP_DIR="${SP_DIR:-${WORKDIR}/runs/demo_cook_simple_SP}"
FCP_DIR="${FCP_DIR:-${K_ROOT}/fcp_training/FCP_demo_cook_simple_SP_vwjkrbm1_20260425-135634}"
ROLLOUT_ENCODER_CKPT="${ROLLOUT_ENCODER_CKPT:-${K_ROOT}/checkpoints/recipe_encoder_ckpt_k6_online_fcp_obs_only}"

DATA_DIR="${DATA_DIR:-${K_ROOT}/data/recipe_data_k6_online_fcp_obs_only_fixed_context}"
NEW_ENCODER_CKPT="${NEW_ENCODER_CKPT:-${K_ROOT}/checkpoints/recipe_encoder_ckpt_k6_online_fcp_obs_only_fixed_context}"

EPISODES_PER_PAIR="${EPISODES_PER_PAIR:-10}"
MAX_STEPS="${MAX_STEPS:-400}"
ENV_MAX_STEPS="${ENV_MAX_STEPS:-400}"
SEGMENT_K="${SEGMENT_K:-6}"
SEGMENT_STRIDE="${SEGMENT_STRIDE:-6}"
SEED="${SEED:-0}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"

cd "${WORKDIR}"

echo "[fixed-context encoder retrain]"
echo "WORKDIR:              ${WORKDIR}"
echo "SP_DIR:               ${SP_DIR}"
echo "FCP_DIR:              ${FCP_DIR}"
echo "ROLLOUT_ENCODER_CKPT: ${ROLLOUT_ENCODER_CKPT}"
echo "DATA_DIR:             ${DATA_DIR}"
echo "NEW_ENCODER_CKPT:     ${NEW_ENCODER_CKPT}"
echo "EPISODES_PER_PAIR:    ${EPISODES_PER_PAIR}"
echo "MAX_STEPS:            ${MAX_STEPS}"
echo "SEGMENT_K:            ${SEGMENT_K}"
echo "SEGMENT_STRIDE:       ${SEGMENT_STRIDE}"
echo

mkdir -p "$(dirname "${DATA_DIR}")" "$(dirname "${NEW_ENCODER_CKPT}")"

echo "[1/2] Collecting fixed-context online dataset"
python -m overcooked_v2_experiments.recipe.collect_online_recipe_data \
  --sp_dir "${SP_DIR}" \
  --fcp_dir "${FCP_DIR}" \
  --encoder_ckpt "${ROLLOUT_ENCODER_CKPT}" \
  --save_dir "${DATA_DIR}" \
  --layout demo_cook_simple \
  --episodes_per_pair "${EPISODES_PER_PAIR}" \
  --max_steps "${MAX_STEPS}" \
  --env_max_steps "${ENV_MAX_STEPS}" \
  --segment_k "${SEGMENT_K}" \
  --segment_stride "${SEGMENT_STRIDE}" \
  --seed "${SEED}"

echo
echo "[2/2] Training fixed-context online encoder"
python -m overcooked_v2_experiments.recipe.train_recipe_encoder_jax \
  --data_dir "${DATA_DIR}" \
  --save_path "${NEW_ENCODER_CKPT}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --seed "${SEED}"

echo
echo "[done]"
echo "Dataset: ${DATA_DIR}"
echo "Encoder: ${NEW_ENCODER_CKPT}"
