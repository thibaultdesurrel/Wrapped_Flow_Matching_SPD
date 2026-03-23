#!/usr/bin/env bash
# Train and evaluate one model per test_group_idx, round-robin across GPUs.
#
# Usage:
#   ./scripts/train_all_groups.sh EXPERIMENT [N_GROUPS] [EXTRA_OVERRIDES...]
#
# Examples:
#   ./scripts/train_all_groups.sh BNCI2014002_diffeocfm 5
#   ./scripts/train_all_groups.sh BNCI2015001_geodesiccfm 2
#   ./scripts/train_all_groups.sh BNCI2014002_diffeocfm 5 trainer.max_epochs=3000

set -euo pipefail
cd "$(dirname "$0")/.."   # cd into runner/

EXPERIMENT="${1:?Usage: $0 EXPERIMENT [N_GROUPS] [EXTRA_OVERRIDES...]}"
N_GROUPS="${2:-5}"
shift 2 2>/dev/null || shift $#  # consume first two args, rest are overrides
EXTRA_OVERRIDES=("$@")

# Detect GPUs
GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' '))
N_GPUS=${#GPUS[@]}

if [ "$N_GPUS" -eq 0 ]; then
    echo "No GPUs found, running on CPU."
    GPUS=(cpu)
    N_GPUS=1
fi

echo "=== Training + evaluating ${EXPERIMENT} with ${N_GROUPS} groups on ${N_GPUS} GPU(s): ${GPUS[*]} ==="

# Train then evaluate a single group. Runs as a background job.
train_and_eval() {
    local group=$1
    local gpu=$2

    echo "[group ${group}] Training on GPU ${gpu}..."
    CUDA_VISIBLE_DEVICES=$gpu uv run src/train.py \
        experiment="$EXPERIMENT" \
        datamodule.test_group_idx="$group" \
        "${EXTRA_OVERRIDES[@]}"

    # Find the run directory that was just created (most recent for this group)
    local run_dir
    run_dir=$(ls -dt logs/"${EXPERIMENT}"/group_"${group}"/*/ 2>/dev/null | head -1)

    if [ -z "$run_dir" ]; then
        echo "[group ${group}] ERROR: Could not find run directory, skipping evaluation."
        return 1
    fi

    echo "[group ${group}] Evaluating ${run_dir}..."
    CUDA_VISIBLE_DEVICES=$gpu uv run src/evaluate.py "$run_dir"
    echo "[group ${group}] Done."
}

PIDS=()
for g in $(seq 0 $((N_GROUPS - 1))); do
    gpu=${GPUS[$((g % N_GPUS))]}
    train_and_eval "$g" "$gpu" &
    PIDS+=($!)
done

echo "=== All ${N_GROUPS} jobs launched (PIDs: ${PIDS[*]}). Waiting... ==="

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: Job with PID $pid failed."
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "=== ${FAILED}/${N_GROUPS} jobs failed. ==="
    exit 1
else
    echo "=== All ${N_GROUPS} jobs completed successfully. ==="
fi
