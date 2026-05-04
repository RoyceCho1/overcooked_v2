#!/usr/bin/env bash
set -euo pipefail

# Run inside Docker from /home/myuser/overcooked_v2_experiments.
# K=2 fixed-context pipeline:
#   collect online dataset -> train encoder -> offline sanity -> online accuracy
#   -> optional FCP training -> return eval.
#
# By default this script runs through online accuracy and stops before FCP
# training if sanity checks fail. To run only selected stages:
#   PIPELINE_STEPS="collect train offline online fcp eval" ./run_k2_fixed_pipeline.sh

WORKDIR="${WORKDIR:-/home/myuser/overcooked_v2_experiments}"
cd "${WORKDIR}"

K_ROOT="${K_ROOT:-${WORKDIR}/runs/k_sweep_demo_cook_simple/k2_fixed}"
DATA_DIR="${DATA_DIR:-${K_ROOT}/data/recipe_data_k2_online_fcp_obs_only_fixed_context}"
ENCODER_CKPT="${ENCODER_CKPT:-${K_ROOT}/checkpoints/recipe_encoder_ckpt_k2_online_fcp_obs_only_fixed_context}"
EVAL_DIR="${EVAL_DIR:-${K_ROOT}/eval}"
ONLINE_DIR="${ONLINE_DIR:-${EVAL_DIR}/online_accuracy}"

SP_DIR="${SP_DIR:-${WORKDIR}/runs/demo_cook_simple_SP}"
ROLLOUT_FCP_DIR="${ROLLOUT_FCP_DIR:-${WORKDIR}/runs/FCP_demo_cook_simple_SP_encoder2}"
ROLLOUT_ENCODER_CKPT="${ROLLOUT_ENCODER_CKPT:-${WORKDIR}/runs/k_sweep_demo_cook_simple/k6/checkpoints/recipe_encoder_ckpt_k6_online_fcp_obs_only_fixed_context}"
FCP_ALIAS="${FCP_ALIAS:-${WORKDIR}/runs/FCP_demo_cook_simple_SP_encoder_k2_fixed}"

COLLECT_EPISODES_PER_PAIR="${COLLECT_EPISODES_PER_PAIR:-10}"
COLLECT_MAX_STEPS="${COLLECT_MAX_STEPS:-400}"
COLLECT_ENV_MAX_STEPS="${COLLECT_ENV_MAX_STEPS:-400}"
SEGMENT_K="${SEGMENT_K:-2}"
SEGMENT_STRIDE="${SEGMENT_STRIDE:-2}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
TRAIN_LR="${TRAIN_LR:-1e-3}"

ONLINE_EPISODES_PER_PAIR="${ONLINE_EPISODES_PER_PAIR:-20}"
ONLINE_MAX_STEPS="${ONLINE_MAX_STEPS:-2000}"
ONLINE_ENV_MAX_STEPS="${ONLINE_ENV_MAX_STEPS:-2000}"

RETURN_EPISODES_PER_PAIR="${RETURN_EPISODES_PER_PAIR:-20}"
RETURN_MAX_STEPS="${RETURN_MAX_STEPS:-400}"
SEED="${SEED:-0}"

PIPELINE_STEPS="${PIPELINE_STEPS:-collect train offline online}"

OFFLINE_JSON="${OFFLINE_JSON:-${EVAL_DIR}/offline_sanity_k2_fixed_val.json}"
OFFLINE_CSV="${OFFLINE_CSV:-${EVAL_DIR}/offline_sanity_k2_fixed_val.csv}"
ONLINE_TIMESTEP_CSV="${ONLINE_TIMESTEP_CSV:-${ONLINE_DIR}/timestep.csv}"
ONLINE_EPISODE_CSV="${ONLINE_EPISODE_CSV:-${ONLINE_DIR}/episode.csv}"
ONLINE_SUMMARY_CSV="${ONLINE_SUMMARY_CSV:-${ONLINE_DIR}/summary.csv}"
RETURN_DETAIL_CSV="${RETURN_DETAIL_CSV:-${WORKDIR}/runs/eval/fcp_encoder_k2_fixed_detail.csv}"
RETURN_SUMMARY_CSV="${RETURN_SUMMARY_CSV:-${WORKDIR}/runs/eval/fcp_encoder_k2_fixed_summary.csv}"

contains_step() {
  local wanted="$1"
  for step in ${PIPELINE_STEPS}; do
    if [[ "${step}" == "${wanted}" ]]; then
      return 0
    fi
  done
  return 1
}

check_metric() {
  local path="$1"
  local metric="$2"
  python - "$path" "$metric" <<'PY'
import csv
import sys
path, metric = sys.argv[1], sys.argv[2]
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("metric") == metric and row.get("group", "global") == "global":
            print(row["value"])
            raise SystemExit(0)
raise SystemExit(f"metric not found: {metric} in {path}")
PY
}

assert_encoder_sanity() {
  python - "$OFFLINE_JSON" "$ONLINE_SUMMARY_CSV" <<'PY'
import csv
import json
import math
import sys

offline_json, online_summary = sys.argv[1], sys.argv[2]
with open(offline_json, encoding="utf-8") as f:
    offline = json.load(f)

if offline["direct_minus_swapped_accuracy"] <= 0:
    raise SystemExit("STOP: offline direct_minus_swapped_accuracy <= 0")
if offline["label_mismatch_suspected"]:
    raise SystemExit("STOP: offline label_mismatch_suspected=1")

metrics = {}
with open(online_summary, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("group") == "global":
            metrics[row["metric"]] = float(row["value"])

required = [
    "mean_first_valid_timestep",
    "label_mismatch_suspected",
    "direct_minus_swapped_accuracy",
    "overall_valid_accuracy",
    "overall_effective_correct_rate",
]
missing = [m for m in required if m not in metrics]
if missing:
    raise SystemExit(f"STOP: missing online metrics: {missing}")
if abs(metrics["mean_first_valid_timestep"] - 2.0) > 1e-6:
    raise SystemExit(f"STOP: mean_first_valid_timestep != 2: {metrics['mean_first_valid_timestep']}")
if metrics["label_mismatch_suspected"] != 0:
    raise SystemExit("STOP: online label_mismatch_suspected=1")
if metrics["direct_minus_swapped_accuracy"] <= 0:
    raise SystemExit("STOP: online direct_minus_swapped_accuracy <= 0")

print("[sanity pass]")
print(f"offline_direct_acc={offline['direct_accuracy']:.6f}")
print(f"offline_direct_minus_swapped={offline['direct_minus_swapped_accuracy']:.6f}")
print(f"online_valid_acc={metrics['overall_valid_accuracy']:.6f}")
print(f"online_effective_correct={metrics['overall_effective_correct_rate']:.6f}")
PY
}

echo "[K=2 fixed-context pipeline]"
echo "WORKDIR:             ${WORKDIR}"
echo "PIPELINE_STEPS:      ${PIPELINE_STEPS}"
echo "K_ROOT:              ${K_ROOT}"
echo "DATA_DIR:            ${DATA_DIR}"
echo "ENCODER_CKPT:        ${ENCODER_CKPT}"
echo "ROLLOUT_FCP_DIR:     ${ROLLOUT_FCP_DIR}"
echo "ROLLOUT_ENCODER:     ${ROLLOUT_ENCODER_CKPT}"
echo "FCP_ALIAS:           ${FCP_ALIAS}"
echo

mkdir -p "${K_ROOT}/data" "${K_ROOT}/checkpoints" "${EVAL_DIR}" "${ONLINE_DIR}" "${WORKDIR}/runs/eval"

if contains_step collect; then
  echo "[1] Collect K=2 fixed-context online dataset"
  python -m overcooked_v2_experiments.recipe.collect_online_recipe_data \
    --sp_dir "${SP_DIR}" \
    --fcp_dir "${ROLLOUT_FCP_DIR}" \
    --encoder_ckpt "${ROLLOUT_ENCODER_CKPT}" \
    --save_dir "${DATA_DIR}" \
    --layout demo_cook_simple \
    --episodes_per_pair "${COLLECT_EPISODES_PER_PAIR}" \
    --max_steps "${COLLECT_MAX_STEPS}" \
    --env_max_steps "${COLLECT_ENV_MAX_STEPS}" \
    --segment_k "${SEGMENT_K}" \
    --segment_stride "${SEGMENT_STRIDE}" \
    --seed "${SEED}"
fi

if contains_step train; then
  echo "[2] Train K=2 encoder"
  python -m overcooked_v2_experiments.recipe.train_recipe_encoder_jax \
    --data_dir "${DATA_DIR}" \
    --save_path "${ENCODER_CKPT}" \
    --epochs "${TRAIN_EPOCHS}" \
    --batch_size "${TRAIN_BATCH_SIZE}" \
    --lr "${TRAIN_LR}" \
    --seed "${SEED}"
fi

if contains_step offline; then
  echo "[3] Offline sanity"
  python -m overcooked_v2_experiments.recipe.eval_recipe_encoder_sanity \
    --data_dir "${DATA_DIR}" \
    --ckpt_path "${ENCODER_CKPT}" \
    --split val \
    --seed "${SEED}" \
    --output_csv "${OFFLINE_CSV}" \
    --output_json "${OFFLINE_JSON}"
fi

if contains_step online; then
  echo "[4] Online accuracy"
  python -m overcooked_v2_experiments.eval.eval_online_recipe_accuracy \
    --sp_dir "${SP_DIR}" \
    --fcp_encoder_dir "${ROLLOUT_FCP_DIR}" \
    --encoder_ckpt "${ENCODER_CKPT}" \
    --layout demo_cook_simple \
    --episodes_per_pair "${ONLINE_EPISODES_PER_PAIR}" \
    --max_steps "${ONLINE_MAX_STEPS}" \
    --env_max_steps "${ONLINE_ENV_MAX_STEPS}" \
    --seed "${SEED}" \
    --output_timestep_csv "${ONLINE_TIMESTEP_CSV}" \
    --output_episode_csv "${ONLINE_EPISODE_CSV}" \
    --output_summary_csv "${ONLINE_SUMMARY_CSV}"
fi

if contains_step fcp; then
  echo "[5] Check sanity before FCP training"
  assert_encoder_sanity

  echo "[6] Train K=2 encoder-conditioned FCP"
  before_file="$(mktemp)"
  after_file="$(mktemp)"
  find "${WORKDIR}/runs" -maxdepth 1 -type d -name 'FCP_demo_cook_simple_SP*' -printf '%f\n' | sort > "${before_file}"

  python -m overcooked_v2_experiments.ppo.main \
    +experiment=rnn-fcp \
    env=demo_cook_simple \
    +FCP="${SP_DIR}" \
    +RECIPE_ENCODER_PATH="${ENCODER_CKPT}" \
    RECIPE_ENCODER_K=2 \
    RECIPE_ENCODER_USE_ACTIONS=false \
    NUM_SEEDS=10 \
    wandb.WANDB_MODE=online \
    wandb.ENTITY="${WANDB_ENTITY:-}" \
    wandb.PROJECT="${WANDB_PROJECT:-}"

  find "${WORKDIR}/runs" -maxdepth 1 -type d -name 'FCP_demo_cook_simple_SP*' -printf '%f\n' | sort > "${after_file}"
  new_run_name="$(comm -13 "${before_file}" "${after_file}" | tail -n 1)"
  rm -f "${before_file}" "${after_file}"
  if [[ -z "${new_run_name}" ]]; then
    echo "Could not infer new FCP run directory. Set FCP_ALIAS manually before eval." >&2
    exit 1
  fi
  new_run="${WORKDIR}/runs/${new_run_name}"
  echo "${new_run}" > "${K_ROOT}/fcp_run_dir.txt"
  if [[ -e "${FCP_ALIAS}" || -L "${FCP_ALIAS}" ]]; then
    if [[ "$(readlink "${FCP_ALIAS}" || true)" == "${new_run}" ]]; then
      echo "FCP alias already points to ${new_run}"
    else
      echo "FCP alias exists and points elsewhere: ${FCP_ALIAS}" >&2
      echo "New run: ${new_run}" >&2
      exit 1
    fi
  else
    ln -s "${new_run}" "${FCP_ALIAS}"
  fi
  echo "FCP run: ${new_run}"
  echo "FCP alias: ${FCP_ALIAS}"
fi

if contains_step eval; then
  echo "[7] Return evaluation"
  python -m overcooked_v2_experiments.eval.compare_fcp_variants \
    --variants encoder \
    --sp_dir "${SP_DIR}" \
    --fcp_encoder_dir "${FCP_ALIAS}" \
    --encoder_ckpt "${ENCODER_CKPT}" \
    --encoder_k 2 \
    --layout demo_cook_simple \
    --episodes_per_pair "${RETURN_EPISODES_PER_PAIR}" \
    --max_steps "${RETURN_MAX_STEPS}" \
    --seed "${SEED}" \
    --output_csv "${RETURN_DETAIL_CSV}" \
    --summary_csv "${RETURN_SUMMARY_CSV}"
fi

echo
echo "[done] K=2 fixed-context pipeline stages completed: ${PIPELINE_STEPS}"
