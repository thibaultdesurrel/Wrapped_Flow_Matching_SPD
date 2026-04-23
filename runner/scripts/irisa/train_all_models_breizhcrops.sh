#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=wgcfm_breizhcrops
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
#SBATCH --array=0-3

set -euo pipefail

MODELS=(
    "breizhcrops_diffeocfm"
    "breizhcrops_geodesiccfm"
    "breizhcrops_wrappedcfm"
    "breizhcrops_wrappedcfm_with_prior"
)

experiment="${MODELS[$SLURM_ARRAY_TASK_ID]}"

module purge
module load pytorch-gpu/py3/2.1.1

mkdir -p job_logs

echo "[${experiment}] Training..."
python src/train.py experiment="$experiment"

# Find the run directory that was just created (most recent for this experiment)
run_dir=$(ls -dt logs/"${experiment}"/*/ 2>/dev/null | head -1)

if [ -z "$run_dir" ]; then
    echo "[${experiment}] ERROR: Could not find run directory, skipping evaluation."
    exit 1
fi

echo "[${experiment}] Evaluating ${run_dir}..."
python src/evaluate.py "$run_dir"
echo "[${experiment}] Done."

