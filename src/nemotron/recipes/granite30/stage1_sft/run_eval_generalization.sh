#!/bin/bash
# =============================================================================
# Generalization Loss Sweep — Granite 8B SFT Checkpoints
# =============================================================================
#
# Loops over checkpoints, launches a separate torchrun per checkpoint.
# Reports cross-entropy loss on held-out reasoning data to detect overfitting.
#
# Usage:
#   bash run_eval_generalization.sh
#
#   # Sample 1000 rows for faster eval:
#   SAMPLE=1000 bash run_eval_generalization.sh
#
#   # Specific steps only:
#   STEPS="100 200 500 1000" bash run_eval_generalization.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Configuration — override via environment variables
# =============================================================================

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite8b_sft_128k_special}"
HELD_OUT_DATA="${HELD_OUT_DATA:-/mnt/vast/proj/checkpoints/bathen/datasets/reasoning_holdout/splits}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/config/eval_granite_8b_generalization.yaml}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/results/granite8b_generalization_loss.json}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-1}"
EVAL_ITERS="${EVAL_ITERS:-}"
SAMPLE="${SAMPLE:-}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
STEPS="${STEPS:-}"

# =============================================================================
# Validate
# =============================================================================

if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: Checkpoint directory not found: ${CHECKPOINT_DIR}"
    exit 1
fi

if [ ! -d "${HELD_OUT_DATA}" ]; then
    echo "ERROR: Held-out data not found: ${HELD_OUT_DATA}"
    exit 1
fi

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config not found: ${CONFIG}"
    exit 1
fi

# =============================================================================
# Discover checkpoints
# =============================================================================

declare -a CKPT_STEPS=()

if [ -n "${STEPS}" ]; then
    for s in ${STEPS}; do
        CKPT_STEPS+=("$s")
    done
else
    for dir in "${CHECKPOINT_DIR}"/iter_*; do
        if [ -d "$dir" ]; then
            step=$(basename "$dir" | sed 's/iter_0*//')
            [ -z "$step" ] && step=0
            CKPT_STEPS+=("$step")
        fi
    done
fi

IFS=$'\n' CKPT_STEPS=($(sort -n <<<"${CKPT_STEPS[*]}")); unset IFS

if [ ${#CKPT_STEPS[@]} -eq 0 ]; then
    echo "ERROR: No checkpoints found in ${CHECKPOINT_DIR}"
    exit 1
fi

# =============================================================================
# Build extra args
# =============================================================================

EXTRA_ARGS=""
if [ -n "${EVAL_ITERS}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --eval-iters ${EVAL_ITERS}"
fi
if [ -n "${SAMPLE}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --sample ${SAMPLE} --sample-seed ${SAMPLE_SEED}"
fi

# =============================================================================
# Run
# =============================================================================

echo "============================================================"
echo " Generalization Loss Sweep — Granite 8B"
echo "============================================================"
echo " Checkpoints:  ${CHECKPOINT_DIR}"
echo " Held-out:     ${HELD_OUT_DATA}"
echo " Config:       ${CONFIG}"
echo " Output:       ${OUTPUT}"
echo " GPUs/node:    ${NPROC_PER_NODE}, Nodes: ${NNODES}"
echo " Steps:        ${CKPT_STEPS[*]}"
echo " Total:        ${#CKPT_STEPS[@]} checkpoints"
[ -n "${SAMPLE}" ] && echo " Sample:       ${SAMPLE} rows (seed=${SAMPLE_SEED})"
echo "============================================================"
echo ""

mkdir -p "$(dirname "${OUTPUT}")"

FAILED_STEPS=()
SUCCEEDED=0

for step in "${CKPT_STEPS[@]}"; do
    echo "--- Step ${step} ---"

    if torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --nnodes="${NNODES}" \
        "${SCRIPT_DIR}/eval_generalization_loss.py" \
        --config "${CONFIG}" \
        --checkpoint-load "${CHECKPOINT_DIR}" \
        --held-out-data "${HELD_OUT_DATA}" \
        --step "${step}" \
        --output "${OUTPUT}" \
        ${EXTRA_ARGS}; then
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        echo "WARNING: Failed for step ${step}"
        FAILED_STEPS+=("$step")
    fi
    echo ""
done

# =============================================================================
# Summary
# =============================================================================

echo "============================================================"
echo " DONE — ${SUCCEEDED}/${#CKPT_STEPS[@]} succeeded"
[ ${#FAILED_STEPS[@]} -gt 0 ] && echo " Failed: ${FAILED_STEPS[*]}"
echo " Results: ${OUTPUT}"
echo "============================================================"

if command -v jq &> /dev/null && [ -f "${OUTPUT}" ]; then
    echo ""
    printf "%-8s | %-12s | %-12s\n" "Step" "Loss" "PPL"
    printf "%-8s-+-%-12s-+-%-12s\n" "--------" "------------" "------------"
    jq -r 'to_entries | sort_by(.key | tonumber) | .[] | "\(.key) \(.value.lm_loss // "N/A") \(.value.lm_loss_ppl // "N/A")"' "${OUTPUT}" | \
    while read -r s l p; do
        printf "%-8s | %-12s | %-12s\n" "$s" "$l" "$p"
    done
fi
