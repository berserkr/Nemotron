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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Configuration
# =============================================================================

CONFIG="${SCRIPT_DIR}/config/eval_granite_8b_generalization.yaml"
HELD_OUT_DATA=/mnt/vast/proj/checkpoints/bathen/datasets/reasoning_holdout/splits
OUTPUT="${SCRIPT_DIR}/results/granite8b_generalization_loss.json"

NPROC_PER_NODE=4
NNODES=1
SAMPLE="${SAMPLE:-}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
EVAL_ITERS="${EVAL_ITERS:-}"

CHECKPOINTS=(
    /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite_v1_sampled_7m_balanced_ash_128k_8b_cp2_fullcot_8500iter/iter_0001000
    /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite_v1_sampled_7m_balanced_ash_128k_8b_cp2_fullcot_8500iter/iter_0002000
    /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite_v1_sampled_7m_balanced_ash_128k_8b_cp2_fullcot_8500iter/iter_0003000
    /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite_v1_sampled_7m_balanced_ash_128k_8b_cp2_fullcot_8500iter/iter_0004000
    /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite_v1_sampled_7m_balanced_ash_128k_8b_cp2_fullcot_8500iter/iter_0005000
    /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite_v1_sampled_7m_balanced_ash_128k_8b_cp2_fullcot_8500iter/iter_0006000
    /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite_v1_sampled_7m_balanced_ash_128k_8b_cp2_fullcot_8500iter/iter_0007000
    /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite_v1_sampled_7m_balanced_ash_128k_8b_cp2_fullcot_8500iter/iter_0008000
    /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite_v1_sampled_7m_balanced_ash_128k_8b_cp2_fullcot_8500iter/iter_0008500
)

# =============================================================================
# Validate
# =============================================================================

if [ ! -d "${HELD_OUT_DATA}" ]; then
    echo "ERROR: Held-out data not found: ${HELD_OUT_DATA}"
    exit 1
fi

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config not found: ${CONFIG}"
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
echo " Held-out:     ${HELD_OUT_DATA}"
echo " Config:       ${CONFIG}"
echo " Output:       ${OUTPUT}"
echo " GPUs/node:    ${NPROC_PER_NODE}, Nodes: ${NNODES}"
echo " Checkpoints:  ${#CHECKPOINTS[@]}"
[ -n "${SAMPLE}" ] && echo " Sample:       ${SAMPLE} rows (seed=${SAMPLE_SEED})"
echo "============================================================"
echo ""

mkdir -p "$(dirname "${OUTPUT}")"

FAILED_STEPS=()
SUCCEEDED=0

for ckpt in "${CHECKPOINTS[@]}"; do
    iter_name=$(basename "$ckpt")
    iter_num="${iter_name#iter_}"
    step=$((10#$iter_num))  # strip leading zeros

    echo "=============================="
    echo "Evaluating: $ckpt"
    echo "Step:       $step"
    echo "=============================="

    if torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --nnodes="${NNODES}" \
        "${SCRIPT_DIR}/eval_generalization_loss.py" \
        --config "${CONFIG}" \
        --checkpoint-load "$(dirname "$ckpt")" \
        --held-out-data "${HELD_OUT_DATA}" \
        --step "${step}" \
        --output "${OUTPUT}" \
        ${EXTRA_ARGS}; then
        SUCCEEDED=$((SUCCEEDED + 1))
        echo "Done: $iter_name"
    else
        echo "WARNING: Failed for $iter_name"
        FAILED_STEPS+=("$step")
    fi
    echo ""
done

# =============================================================================
# Summary
# =============================================================================

echo "============================================================"
echo " DONE — ${SUCCEEDED}/${#CHECKPOINTS[@]} succeeded"
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

echo ""
echo "All evaluations complete."
