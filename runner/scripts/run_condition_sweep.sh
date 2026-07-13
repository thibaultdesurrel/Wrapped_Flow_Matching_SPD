#!/usr/bin/env bash
# Condition-number sweep on the WG-mixture benchmark: for each kappa, train +
# evaluate all 4 CFM models x N_SEEDS seeds, round-robin across GPUs (CPU
# fallback). Log dirs embed kappa so different kappas never shadow each other:
#   logs/wg_mixture_k<kappa>_<model>/seed_<seed>/<timestamp>
# which is exactly the layout make_synthetic_table.py aggregates
# (dataset "wg_mixture_k<kappa>", model "<model>").
#
# Usage:
#   ./scripts/run_condition_sweep.sh                       # kappa in {10,100,1000,10000}, 5 seeds
#   ./scripts/run_condition_sweep.sh "10000" 2             # quick signal check at kappa=1e4
#   ./scripts/run_condition_sweep.sh "100 10000" 3 trainer.max_epochs=2000
#
# Set JOBS_PER_GPU to run more than one job per GPU concurrently, e.g.:
#   JOBS_PER_GPU=2 ./scripts/run_condition_sweep.sh        # 2 jobs per GPU in parallel
#
# Afterwards:
#   uv run scripts/make_synthetic_table.py   # -> results/synthetic_summary.csv

set -euo pipefail
cd "$(dirname "$0")/.."   # cd into runner/

KAPPAS_STR="${1:-10 100 1000 10000}"
N_SEEDS="${2:-5}"
EXTRA_OVERRIDES=()
if [ "$#" -gt 2 ]; then
    shift 2
    EXTRA_OVERRIDES=("$@")
fi

read -r -a KAPPAS <<< "$KAPPAS_STR"
MODELS=(wrappedcfm geodesiccfm diffeocfm euclideancfm)

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

echo "=== Condition-number sweep (WG mixture) ==="
echo "Kappas: ${KAPPAS[*]}"
echo "Models: ${MODELS[*]}"
echo "Seeds:  ${SEEDS[*]}"
echo "GPUs:   ${GPUS[*]} (${JOBS_PER_GPU} job(s) per GPU, ${N_SLOTS} slot(s) total)"

# Train then evaluate a single (kappa, model, seed) run. Background job.
train_and_eval() {
    local kappa=$1
    local model=$2
    local seed=$3
    local gpu=$4

    local run_dir="logs/wg_mixture_k${kappa}_${model}/seed_${seed}/$(date +%Y-%m-%d_%H-%M-%S)"
    echo "[k=${kappa} ${model} seed=${seed}] Training on GPU ${gpu} -> ${run_dir}"
    CUDA_VISIBLE_DEVICES=$gpu uv run src/train.py \
        experiment="wg_mixture_${model}" \
        seed="$seed" \
        datamodule.kappa="$kappa" \
        hydra.run.dir="$run_dir" \
        ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}

    echo "[k=${kappa} ${model} seed=${seed}] Evaluating ${run_dir}..."
    CUDA_VISIBLE_DEVICES=$gpu uv run src/evaluate_synthetic.py "$run_dir"
    echo "[k=${kappa} ${model} seed=${seed}] Done."
}

# Build the full job list (kappa x model x seed).
JOBS=()
for kappa in "${KAPPAS[@]}"; do
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            JOBS+=("${kappa}:${model}:${seed}")
        done
    done
done

# Continuous scheduler: keep at most one job per slot running (same scheme as
# run_synthetic_benchmark.sh; indexed arrays so it works on bash 3.2 / macOS).
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
            IFS=":" read -r kappa model seed <<< "${JOBS[$job_idx]}"
            train_and_eval "$kappa" "$model" "$seed" "${SLOT_GPU[$s]}" &
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
