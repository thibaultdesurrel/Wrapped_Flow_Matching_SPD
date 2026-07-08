#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=wgcfm_textile
#SBATCH --chdir=/lustre/fswork/projects/rech/ycx/utq99jv/tibo_aimi/SPD_WFM_2/runner
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=11
#SBATCH --hint=nomultithread
#SBATCH --time=6:00:00
#SBATCH --output=job_logs/%x_%a.out
#SBATCH --error=job_logs/%x_%a.err
#SBATCH --partition=gpu_p13
#SBATCH -A ycx@v100
#SBATCH --array=0-19
# Job layout (4 models x 5 seeds, flattened):
#   model = TASK_ID / 5,  seed = TASK_ID % 5

set -euo pipefail

N_SEEDS=5

MODELS=(
    "textile_euclideancfm"
    "textile_diffeocfm"
    "textile_geodesiccfm"
    "textile_wrappedcfm"
)

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
experiment="${MODELS[$MODEL_IDX]}"

module purge
module load pytorch-gpu/py3/2.1.1

mkdir -p job_logs

echo "[${experiment} seed=${SEED}] Training..."
python src/train.py experiment="$experiment" seed="$SEED"

# Find the run directory that was just created (most recent for this experiment/seed)
run_dir=$(ls -dt logs/"${experiment}"/seed_"${SEED}"/*/ 2>/dev/null | head -1)

if [ -z "$run_dir" ]; then
    echo "[${experiment} seed=${SEED}] ERROR: Could not find run directory, skipping evaluation."
    exit 1
fi

echo "[${experiment} seed=${SEED}] Evaluating ${run_dir}..."
python src/evaluate.py "$run_dir"
echo "[${experiment} seed=${SEED}] Done."

