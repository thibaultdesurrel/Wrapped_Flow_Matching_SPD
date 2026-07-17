#!/usr/bin/env bash
# Train + evaluate the synthetic CFM benchmark: all (experiment x seed) runs,
# round-robin across GPUs (or CPU fallback). Each seed is written to its own
# subdir (logs/<experiment>/seed_<seed>/<timestamp>) so fetch_results.py
# aggregates all seeds (it keeps the latest timestamp *per group*).
#
# Usage:
#   ./scripts/run_synthetic_benchmark.sh                       # full 8 exp x 5 seeds
#   ./scripts/run_synthetic_benchmark.sh EXPERIMENT N_SEEDS [EXTRA_OVERRIDES...]
#
# Examples:
#   ./scripts/run_synthetic_benchmark.sh                                  # full grid
#   ./scripts/run_synthetic_benchmark.sh figure8_euclideancfm 1 trainer.max_epochs=5   # smoke test
#
# Set JOBS_PER_GPU to run more than one job per GPU concurrently, e.g.:
#   JOBS_PER_GPU=2 ./scripts/run_synthetic_benchmark.sh        # 2 jobs per GPU in parallel
#
# After this finishes:
#   uv run scripts/fetch_results.py          # -> results/all_results.csv
#   uv run scripts/make_synthetic_table.py   # -> results/synthetic_summary.csv

set -euo pipefail
cd "$(dirname "$0")/.."   # cd into runner/

# --- Parse args: full grid by default, or a single (experiment, n_seeds) run ---
DATASETS=(figure8 swiss_roll swiss_roll_dim_5 swiss_roll_dim_10)
MODELS=(wrappedcfm geodesiccfm diffeocfm euclideancfm)

EXPERIMENTS=()
N_SEEDS=5
EXTRA_OVERRIDES=()

if [ "$#" -eq 0 ]; then
    for ds in "${DATASETS[@]}"; do
        for m in "${MODELS[@]}"; do
            EXPERIMENTS+=("${ds}_${m}")
        done
    done
else
    EXPERIMENTS=("${1:?Usage: $0 [EXPERIMENT N_SEEDS [EXTRA_OVERRIDES...]]}")
    N_SEEDS="${2:?Usage: $0 [EXPERIMENT N_SEEDS [EXTRA_OVERRIDES...]]}"
    shift 2
    EXTRA_OVERRIDES=("$@")
fi

SEEDS=()
for i in $(seq 0 $((N_SEEDS - 1))); do
    SEEDS+=("$i")
done

# --- Detect GPUs (CPU fallback) ---
GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' || true))
N_GPUS=${#GPUS[@]}
if [ "$N_GPUS" -eq 0 ]; then
    echo "No GPUs found, running on CPU."
    GPUS=(cpu)
    N_GPUS=1
fi

# --- Slots: JOBS_PER_GPU concurrent jobs per GPU (default 1). Each slot maps
# to one GPU; with JOBS_PER_GPU=2 each GPU id appears in two slots. ---
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
SLOT_GPU=()
for ((r = 0; r < JOBS_PER_GPU; r++)); do
    for gpu in "${GPUS[@]}"; do
        SLOT_GPU+=("$gpu")
    done
done
N_SLOTS=${#SLOT_GPU[@]}

echo "=== Synthetic benchmark ==="
echo "Experiments: ${EXPERIMENTS[*]}"
echo "Seeds:       ${SEEDS[*]}"
echo "GPUs:        ${GPUS[*]} (${JOBS_PER_GPU} job(s) per GPU, ${N_SLOTS} slot(s) total)"

# Train then evaluate a single (experiment, seed) run. Background job.
train_and_eval() {
    local exp=$1
    local seed=$2
    local gpu=$3

    local run_dir="logs/${exp}/seed_${seed}/$(date +%Y-%m-%d_%H-%M-%S)"
    echo "[${exp} seed=${seed}] Training on GPU ${gpu} -> ${run_dir}"
    CUDA_VISIBLE_DEVICES=$gpu uv run src/train.py \
        experiment="$exp" \
        seed="$seed" \
        hydra.run.dir="$run_dir" \
        "${EXTRA_OVERRIDES[@]}"

    echo "[${exp} seed=${seed}] Evaluating ${run_dir}..."
    CUDA_VISIBLE_DEVICES=$gpu uv run src/evaluate_synthetic.py "$run_dir"
    echo "[${exp} seed=${seed}] Done."
}

# Build the full job list (experiment x seed).
JOBS=()
for exp in "${EXPERIMENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        JOBS+=("${exp}:${seed}")
    done
done

# Continuous scheduler: keep at most one job per slot running (same scheme as
# run_condition_sweep.sh; indexed arrays so it works on bash 3.2 / macOS). As
# soon as a slot frees up it picks the next job, so GPUs never idle waiting
# for a slow batch mate (figure8 runs are much shorter than swiss_roll).
echo "=== ${#JOBS[@]} jobs total, up to ${N_SLOTS} running in parallel. ==="
SLOT_PID=()
for ((s = 0; s < N_SLOTS; s++)); do SLOT_PID[$s]=""; done

FAILED=0
job_idx=0
running=0

while [ "$job_idx" -lt "${#JOBS[@]}" ] || [ "$running" -gt 0 ]; do
    # Reap finished jobs and free their slot.
    for ((s = 0; s < N_SLOTS; s++)); do
        pid=${SLOT_PID[$s]}
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            if ! wait "$pid"; then
                echo "ERROR: Job with PID $pid failed."
                FAILED=$((FAILED + 1))
            fi
            SLOT_PID[$s]=""
            running=$((running - 1))
        fi
    done

    # Launch new jobs onto free slots.
    for ((s = 0; s < N_SLOTS; s++)); do
        [ "$job_idx" -lt "${#JOBS[@]}" ] || break
        if [ -z "${SLOT_PID[$s]}" ]; then
            IFS=":" read -r exp seed <<< "${JOBS[$job_idx]}"
            train_and_eval "$exp" "$seed" "${SLOT_GPU[$s]}" &
            SLOT_PID[$s]=$!
            running=$((running + 1))
            job_idx=$((job_idx + 1))
        fi
    done

    sleep 2
done

if [ "$FAILED" -gt 0 ]; then
    echo "=== ${FAILED}/${#JOBS[@]} jobs failed. ==="
    exit 1
else
    echo "=== All ${#JOBS[@]} jobs completed successfully. ==="
fi
