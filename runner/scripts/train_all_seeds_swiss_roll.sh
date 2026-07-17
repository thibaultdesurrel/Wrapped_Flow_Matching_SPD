#!/usr/bin/env bash
# Train and evaluate one model per seed on a swiss_roll experiment, round-robin across GPUs.
#
# Usage:
#   ./scripts/train_all_seeds_swiss_roll.sh EXPERIMENT N_SEEDS [EXTRA_OVERRIDES...]
#
# Examples:
#   ./scripts/train_all_seeds_swiss_roll.sh swiss_roll_diffeocfm 5
#   ./scripts/train_all_seeds_swiss_roll.sh swiss_roll_geodesiccfm 3 trainer.max_epochs=3000
#   ./scripts/train_all_seeds_swiss_roll.sh swiss_roll_wrappedcfm 4 42 123 7 99

set -euo pipefail
cd "$(dirname "$0")/.."   # cd into runner/

EXPERIMENT="${1:?Usage: $0 EXPERIMENT N_SEEDS [EXTRA_OVERRIDES...]}"
N_SEEDS="${2:?Usage: $0 EXPERIMENT N_SEEDS [EXTRA_OVERRIDES...]}"
shift 2

# Generate default seeds 0..N_SEEDS-1, or accept explicit seeds as remaining args
# if remaining args are all integers, treat them as seeds; otherwise treat as overrides
SEEDS=()
EXTRA_OVERRIDES=()

for arg in "$@"; do
    if [[ "$arg" =~ ^[0-9]+$ ]]; then
        SEEDS+=("$arg")
    else
        EXTRA_OVERRIDES+=("$arg")
    fi
done

if [ "${#SEEDS[@]}" -eq 0 ]; then
    for i in $(seq 0 $((N_SEEDS - 1))); do
        SEEDS+=("$i")
    done
elif [ "${#SEEDS[@]}" -ne "$N_SEEDS" ]; then
    echo "ERROR: N_SEEDS=${N_SEEDS} but ${#SEEDS[@]} explicit seeds provided: ${SEEDS[*]}"
    exit 1
fi

# Detect GPUs
GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' '))
N_GPUS=${#GPUS[@]}

if [ "$N_GPUS" -eq 0 ]; then
    echo "No GPUs found, running on CPU."
    GPUS=(cpu)
    N_GPUS=1
fi

echo "=== Training + evaluating ${EXPERIMENT} with seeds [${SEEDS[*]}] on ${N_GPUS} GPU(s): ${GPUS[*]} ==="

# Train then evaluate a single seed. Runs as a background job.
train_and_eval() {
    local seed=$1
    local gpu=$2

    echo "[seed ${seed}] Training on GPU ${gpu}..."
    CUDA_VISIBLE_DEVICES=$gpu uv run src/train.py \
        experiment="$EXPERIMENT" \
        seed="$seed" \
        "${EXTRA_OVERRIDES[@]}"

    # Find the run directory that was just created (most recent for this experiment)
    local run_dir
    run_dir=$(ls -dt logs/"${EXPERIMENT}"/*/ 2>/dev/null | head -1)

    if [ -z "$run_dir" ]; then
        echo "[seed ${seed}] ERROR: Could not find run directory, skipping evaluation."
        return 1
    fi

    echo "[seed ${seed}] Evaluating ${run_dir}..."
    CUDA_VISIBLE_DEVICES=$gpu uv run src/evaluate.py "$run_dir"
    echo "[seed ${seed}] Done."
}

PIDS=()
for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    gpu=${GPUS[$((i % N_GPUS))]}
    train_and_eval "$seed" "$gpu" &
    PIDS+=($!)
done

echo "=== All ${N_SEEDS} jobs launched (PIDs: ${PIDS[*]}). Waiting... ==="

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: Job with PID $pid failed."
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "=== ${FAILED}/${N_SEEDS} jobs failed. ==="
    exit 1
else
    echo "=== All ${N_SEEDS} jobs completed successfully. ==="
fi
