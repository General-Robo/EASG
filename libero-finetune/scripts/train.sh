#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${RECIPE:-}" ]]; then
  case "$RECIPE" in
    libero_spatial|libero_object|libero_goal|libero_10) source "$PROJECT_ROOT/recipes/$RECIPE.sh" ;;
    *) echo "Unknown recipe: $RECIPE" >&2; exit 1 ;;
  esac
fi
SUITE="${SUITE:-libero_10}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/training_data/$SUITE}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.5-0.8B}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_ROOT/samples.jsonl}"
EVAL_FILE="${EVAL_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/$SUITE}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
INIT_ADAPTER_FROM="${INIT_ADAPTER_FROM:-}"
ANSWER_ONLY_LOGITS="${ANSWER_ONLY_LOGITS:-true}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a VISIBLE_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${NUM_GPUS:-${#VISIBLE_GPU_LIST[@]}}"

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-8192}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-90}"
MAX_STEPS="${MAX_STEPS:--1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"

CE_WEIGHT="${CE_WEIGHT:-5.0}"
GRIPPER_CE_WEIGHT="${GRIPPER_CE_WEIGHT:-5.0}"
SIGN_FORMAT_WEIGHT="${SIGN_FORMAT_WEIGHT:-1.0}"
DIGIT_FORMAT_WEIGHT="${DIGIT_FORMAT_WEIGHT:-1.0}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj,attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2,merger.linear_fc1,merger.linear_fc2}"

export CUDA_VISIBLE_DEVICES

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "Training file not found: $TRAIN_FILE" >&2
  exit 1
fi
if [[ "$NUM_GPUS" -ne "${#VISIBLE_GPU_LIST[@]}" ]]; then
  echo "NUM_GPUS=$NUM_GPUS does not match CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

echo "Dataset: $TRAIN_FILE"
echo "GPUs: $CUDA_VISIBLE_DEVICES (count=$NUM_GPUS)"
echo "Epochs: $NUM_TRAIN_EPOCHS"
echo "Output: $OUTPUT_DIR"

args=(
  "$PROJECT_ROOT/scripts/finetune.py"
  --model-path "$MODEL_PATH"
  --train-file "$TRAIN_FILE"
  --image-root "$DATA_ROOT"
  --output-dir "$OUTPUT_DIR"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --num-train-epochs "$NUM_TRAIN_EPOCHS"
  --max-steps "$MAX_STEPS"
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning-rate "$LEARNING_RATE"
  --weight-decay 0.0
  --warmup-ratio 0.04
  --lr-scheduler-type cosine
  --max-grad-norm 1.0
  --ce-weight "$CE_WEIGHT"
  --gripper-ce-weight "$GRIPPER_CE_WEIGHT"
  --sign-format-weight "$SIGN_FORMAT_WEIGHT"
  --digit-format-weight "$DIGIT_FORMAT_WEIGHT"
  --lora-r "$LORA_R"
  --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT"
  --target-modules "$TARGET_MODULES"
  --attn-implementation sdpa
  --gradient-checkpointing
  --bf16
  --optim adamw_torch
  --logging-steps "$LOGGING_STEPS"
  --save-steps "$SAVE_STEPS"
  --save-total-limit "$SAVE_TOTAL_LIMIT"
  --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
)

[[ -n "$INIT_ADAPTER_FROM" ]] && args+=(--init-adapter-from "$INIT_ADAPTER_FROM")
[[ "$ANSWER_ONLY_LOGITS" == "false" ]] && args+=(--no-answer-only-logits)

[[ -n "$EVAL_FILE" ]] && args+=(--eval-file "$EVAL_FILE")
[[ -n "$RESUME_FROM_CHECKPOINT" ]] && args+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
[[ -n "$ADAPTER_PATH" ]] && args+=(--adapter-path "$ADAPTER_PATH")

if [[ "$NUM_GPUS" -gt 1 ]]; then
  exec "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    "${args[@]}"
fi

exec "$PYTHON_BIN" "${args[@]}"
