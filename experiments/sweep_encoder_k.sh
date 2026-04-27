#!/usr/bin/env bash
set -euo pipefail

# K-sweep pipeline for encoder-conditioned FCP on demo_cook_simple.
#
# Stages per K:
#   1) collect recipe dataset (ego obs-only)
#   2) train recipe encoder
#   3) train FCP encoder-conditioned policy
#   4) evaluate encoder variant vs SP partners
#
# Resume behavior:
# - If dataset_meta.json exists, data collection is skipped (unless FORCE_RECOLLECT=1).
# - If encoder ckpt dir exists, encoder training is skipped (unless FORCE_RETRAIN_ENCODER=1).
# - If fcp_run_dir.txt points to a valid run dir with ckpt_final files, FCP training is skipped
#   (unless FORCE_RETRAIN_FCP=1).
# - Evaluation uses compare_fcp_variants.py resume logic via output CSV.
#
# Usage:
#   ./sweep_encoder_k.sh <SP_RUN_DIR> [K_LIST]
#
# Example:
#   ./sweep_encoder_k.sh /home/myuser/overcooked_v2_experiments/runs/demo_cook_simple_SP "2 4 6 8 10"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <SP_RUN_DIR> [K_LIST]"
  echo "Example: $0 /home/myuser/overcooked_v2_experiments/runs/demo_cook_simple_SP \"2 4 6 8 10\""
  exit 1
fi

SP_RUN_DIR="$(realpath "$1")"
K_LIST="${2:-2 4 6 8 10}"

if [[ ! -d "$SP_RUN_DIR" ]]; then
  echo "Error: SP_RUN_DIR not found: $SP_RUN_DIR"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SWEEP_ROOT="${SWEEP_ROOT:-$SCRIPT_DIR/runs/k_sweep_demo_cook_simple}"
mkdir -p "$SWEEP_ROOT"

# Training/eval defaults (override via env vars).
ENC_EPOCHS="${ENC_EPOCHS:-50}"
ENC_BATCH_SIZE="${ENC_BATCH_SIZE:-32}"
ENC_LR="${ENC_LR:-1e-3}"
ENC_SEED="${ENC_SEED:-0}"

FCP_NUM_SEEDS="${FCP_NUM_SEEDS:-10}"
FCP_TOTAL_TIMESTEPS="${FCP_TOTAL_TIMESTEPS:-3000000}"
FCP_WANDB_MODE="${FCP_WANDB_MODE:-online}"

EVAL_EPISODES="${EVAL_EPISODES:-10}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-400}"
EVAL_SEED="${EVAL_SEED:-0}"

FORCE_RECOLLECT="${FORCE_RECOLLECT:-0}"
FORCE_RETRAIN_ENCODER="${FORCE_RETRAIN_ENCODER:-0}"
FORCE_RETRAIN_FCP="${FORCE_RETRAIN_FCP:-0}"

echo "[sweep] SP_RUN_DIR: $SP_RUN_DIR"
echo "[sweep] K_LIST: $K_LIST"
echo "[sweep] SWEEP_ROOT: $SWEEP_ROOT"

for K in $K_LIST; do
  echo ""
  echo "======================================================="
  echo "[sweep] K=$K"
  echo "======================================================="

  K_DIR="$SWEEP_ROOT/k${K}"
  DATA_DIR="$K_DIR/recipe_data_k${K}"
  ENCODER_CKPT="$K_DIR/recipe_encoder_ckpt_k${K}"
  FCP_RUN_PTR="$K_DIR/fcp_run_dir.txt"
  EVAL_DETAIL="$K_DIR/fcp_encoder_detail_k${K}.csv"
  EVAL_SUMMARY="$K_DIR/fcp_encoder_summary_k${K}.csv"

  mkdir -p "$K_DIR"

  # -----------------------------------------------------------
  # 1) Collect dataset
  # -----------------------------------------------------------
  if [[ "$FORCE_RECOLLECT" == "1" || ! -f "$DATA_DIR/dataset_meta.json" ]]; then
    echo "[K=$K] Collecting dataset -> $DATA_DIR"
    RECIPE_SEGMENT_K="$K" \
    RECIPE_SEGMENT_STRIDE="$K" \
    RECIPE_MAX_STEPS="$EVAL_MAX_STEPS" \
    RECIPE_EPISODES_PER_PARTNER=50 \
    RECIPE_INCLUDE_PARTNER_ACTION=false \
    RECIPE_PARTNER_ACTION_VISIBILITY_AWARE=false \
      ./collect_recipe_dataset.sh "$SP_RUN_DIR" "$DATA_DIR"
  else
    echo "[K=$K] Dataset exists, skip: $DATA_DIR"
  fi

  # -----------------------------------------------------------
  # 2) Train encoder
  # -----------------------------------------------------------
  if [[ "$FORCE_RETRAIN_ENCODER" == "1" || ! -d "$ENCODER_CKPT" || ! -f "${ENCODER_CKPT}.meta.json" ]]; then
    echo "[K=$K] Training encoder -> $ENCODER_CKPT"
    python -m overcooked_v2_experiments.recipe.train_recipe_encoder_jax \
      --data_dir "$DATA_DIR" \
      --save_path "$ENCODER_CKPT" \
      --epochs "$ENC_EPOCHS" \
      --batch_size "$ENC_BATCH_SIZE" \
      --lr "$ENC_LR" \
      --seed "$ENC_SEED"
  else
    echo "[K=$K] Encoder ckpt exists, skip: $ENCODER_CKPT"
  fi

  # -----------------------------------------------------------
  # 3) Train FCP encoder-conditioned
  # -----------------------------------------------------------
  FCP_RUN_DIR=""
  if [[ -f "$FCP_RUN_PTR" ]]; then
    CANDIDATE="$(cat "$FCP_RUN_PTR" || true)"
    if [[ -n "$CANDIDATE" && -d "$CANDIDATE" ]]; then
      CKPT_COUNT="$(find "$CANDIDATE" -maxdepth 2 -type d -path '*/run_*/ckpt_final' | wc -l | tr -d ' ')"
      if [[ "$CKPT_COUNT" != "0" ]]; then
        FCP_RUN_DIR="$CANDIDATE"
      fi
    fi
  fi

  if [[ "$FORCE_RETRAIN_FCP" == "1" || -z "$FCP_RUN_DIR" ]]; then
    echo "[K=$K] Training FCP (encoder-conditioned)"
    python -m overcooked_v2_experiments.ppo.main \
      +experiment=rnn-fcp env=demo_cook_simple \
      +OPTIONAL_PREFIX="k_sweep_demo_cook_simple/k${K}" \
      ++FCP="$SP_RUN_DIR" \
      ++RECIPE_ENCODER_PATH="$ENCODER_CKPT" \
      ++RECIPE_ENCODER_K="$K" \
      NUM_SEEDS="$FCP_NUM_SEEDS" \
      model.TOTAL_TIMESTEPS="$FCP_TOTAL_TIMESTEPS" \
      wandb.WANDB_MODE="$FCP_WANDB_MODE" \
      wandb.ENTITY="${WANDB_ENTITY:-}" \
      wandb.PROJECT="${WANDB_PROJECT:-}"

    PREFIX_DIR="$SCRIPT_DIR/runs/k_sweep_demo_cook_simple/k${K}"
    LATEST_RUN="$(ls -td "$PREFIX_DIR"/FCP_* 2>/dev/null | head -n 1 || true)"
    if [[ -z "$LATEST_RUN" ]]; then
      echo "[K=$K] Error: could not find latest FCP run under $PREFIX_DIR"
      exit 1
    fi
    FCP_RUN_DIR="$(realpath "$LATEST_RUN")"
    echo "$FCP_RUN_DIR" > "$FCP_RUN_PTR"
    echo "[K=$K] FCP run dir saved: $FCP_RUN_DIR"
  else
    echo "[K=$K] Reusing FCP run dir from pointer: $FCP_RUN_DIR"
  fi

  # -----------------------------------------------------------
  # 4) Evaluate encoder variant (resume supported)
  # -----------------------------------------------------------
  echo "[K=$K] Evaluating encoder variant -> $EVAL_SUMMARY"
  python -m overcooked_v2_experiments.eval.compare_fcp_variants \
    --variants encoder \
    --sp_dir "$SP_RUN_DIR" \
    --fcp_encoder_dir "$FCP_RUN_DIR" \
    --encoder_ckpt "$ENCODER_CKPT" \
    --layout demo_cook_simple \
    --episodes_per_pair "$EVAL_EPISODES" \
    --max_steps "$EVAL_MAX_STEPS" \
    --seed "$EVAL_SEED" \
    --output_csv "$EVAL_DETAIL" \
    --summary_csv "$EVAL_SUMMARY"

  echo "[K=$K] done."
done

echo ""
echo "[sweep] All requested K values finished."
echo "[sweep] Results root: $SWEEP_ROOT"
