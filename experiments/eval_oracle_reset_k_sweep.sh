#!/usr/bin/env bash
set -euo pipefail

# Run inside Docker from /home/myuser/overcooked_v2_experiments.
# Evaluates oracle-trained FCP with oracle context hidden for K steps after
# episode reset / recipe change. This isolates blind-window sensitivity.

WORKDIR="${WORKDIR:-/home/myuser/overcooked_v2_experiments}"
SP_DIR="${SP_DIR:-${WORKDIR}/runs/demo_cook_simple_SP}"
FCP_ORACLE_DIR="${FCP_ORACLE_DIR:-${WORKDIR}/runs/FCP_demo_cook_simple_SP_oracle}"
EVAL_DIR="${EVAL_DIR:-${WORKDIR}/runs/eval}"
K_LIST="${K_LIST:-2 3 4}"
EPISODES_PER_PAIR="${EPISODES_PER_PAIR:-20}"
MAX_STEPS="${MAX_STEPS:-400}"
SEED="${SEED:-0}"

cd "${WORKDIR}"
mkdir -p "${EVAL_DIR}"

echo "[oracle reset-k sweep]"
echo "SP_DIR:            ${SP_DIR}"
echo "FCP_ORACLE_DIR:    ${FCP_ORACLE_DIR}"
echo "EVAL_DIR:          ${EVAL_DIR}"
echo "K_LIST:            ${K_LIST}"
echo "EPISODES_PER_PAIR: ${EPISODES_PER_PAIR}"
echo "MAX_STEPS:         ${MAX_STEPS}"
echo

for K in ${K_LIST}; do
  echo "======================================================="
  echo "[oracle_reset_k${K}]"
  echo "======================================================="
  python -m overcooked_v2_experiments.eval.compare_fcp_variants \
    --variants oracle_reset_k \
    --sp_dir "${SP_DIR}" \
    --fcp_oracle_dir "${FCP_ORACLE_DIR}" \
    --oracle_reset_k "${K}" \
    --layout demo_cook_simple \
    --episodes_per_pair "${EPISODES_PER_PAIR}" \
    --max_steps "${MAX_STEPS}" \
    --seed "${SEED}" \
    --output_csv "${EVAL_DIR}/fcp_oracle_reset_k${K}_detail.csv" \
    --summary_csv "${EVAL_DIR}/fcp_oracle_reset_k${K}_summary.csv"
done

echo
echo "[done] results saved under ${EVAL_DIR}"
