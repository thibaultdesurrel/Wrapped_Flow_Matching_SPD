#!/usr/bin/env bash
# Train and evaluate all 4 models on the BreizhCrops dataset, spread across 2 GPUs.
#
# Usage:
#   ./scripts/train_all_models_breizhcrops.sh [EXTRA_OVERRIDES...]
#
# Examples:
#   ./scripts/train_all_models_breizhcrops.sh
#   ./scripts/train_all_models_breizhcrops.sh trainer.max_epochs=1000

set -euo pipefail
cd "$(dirname "$0")/.."   # cd into runner/

EXTRA_OVERRIDES=("$@")

MODELS=(
    "breizhcrops_diffeocfm"
    "breizhcrops_geodesiccfm"
    "breizhcrops_wrappedcfm"
    "breizhcrops_wrappedcfm_with_prior"
)

# Detect GPUs
GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' '))
N_GPUS=${#GPUS[@]}

if [ "$N_GPUS" -eq 0 ]; then
    echo "No GPUs found, running on CPU."
    GPUS=(cpu)
    N_GPUS=1
fi

echo "=== Training + evaluating ${#MODELS[@]} models on BreizhCrops on ${N_GPUS} GPU(s): ${GPUS[*]} ==="

# Train then evaluate a single model. Runs as a background job.
train_and_eval() {
    local experiment=$1
    local gpu=$2

    echo "[${experiment}] Training on GPU ${gpu}..."
    CUDA_VISIBLE_DEVICES=$gpu uv run src/train.py \
        experiment="$experiment" \
        "${EXTRA_OVERRIDES[@]}"

    # Find the run directory that was just created (most recent for this experiment)
    local run_dir
    run_dir=$(ls -dt logs/"${experiment}"/*/ 2>/dev/null | head -1)

    if [ -z "$run_dir" ]; then
        echo "[${experiment}] ERROR: Could not find run directory, skipping evaluation."
        return 1
    fi

    echo "[${experiment}] Evaluating ${run_dir}..."
    CUDA_VISIBLE_DEVICES=$gpu uv run src/evaluate.py "$run_dir"
    echo "[${experiment}] Done."
}

PIDS=()
for i in "${!MODELS[@]}"; do
    experiment="${MODELS[$i]}"
    gpu=${GPUS[$((i % N_GPUS))]}
    train_and_eval "$experiment" "$gpu" &
    PIDS+=($!)
done

echo "=== All ${#MODELS[@]} jobs launched (PIDs: ${PIDS[*]}). Waiting... ==="

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: Job with PID $pid failed."
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "=== ${FAILED}/${#MODELS[@]} jobs failed. ==="
    exit 1
else
    echo "=== All ${#MODELS[@]} jobs completed successfully. ==="
fi
