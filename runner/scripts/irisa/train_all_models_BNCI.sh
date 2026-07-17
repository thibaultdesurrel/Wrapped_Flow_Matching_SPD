#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=wgcfm_bnci
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
#SBATCH --array=0-27
#
# Job layout (4 models x n_groups, flattened):
#   BNCI2014002 : indices  0-19  (4 models x 5 groups)
#   BNCI2015001 : indices 20-27  (4 models x 2 groups)

set -euo pipefail

MNE_DATA_PATH="/lustre/fswork/projects/rech/ycx/utq99jv/mne_data"

MODELS=(
    "euclideancfm"
    "diffeocfm"
    "geodesiccfm"
    "wrappedcfm"
)

TASK_ID=$SLURM_ARRAY_TASK_ID

# Decode dataset, model and group from the flat task index
if [ "$TASK_ID" -lt 20 ]; then
    DATASET="BNCI2014002"
    N_GROUPS=5
    OFFSET=0
else
    DATASET="BNCI2015001"
    N_GROUPS=2
    OFFSET=20
fi

MODEL_IDX=$(( (TASK_ID - OFFSET) / N_GROUPS ))
GROUP_IDX=$(( (TASK_ID - OFFSET) % N_GROUPS ))
EXPERIMENT="${DATASET}_${MODELS[$MODEL_IDX]}"

module purge
module load pytorch-gpu/py3/2.1.1

mkdir -p job_logs

echo "[${EXPERIMENT} group ${GROUP_IDX}] Training..."
python src/train.py \
    experiment="$EXPERIMENT" \
    datamodule.test_group_idx="$GROUP_IDX" \
    datamodule.mne_data_path="$MNE_DATA_PATH"

# Find the run directory that was just created (most recent for this experiment/group)
run_dir=$(ls -dt logs/"${EXPERIMENT}"/group_"${GROUP_IDX}"/*/ 2>/dev/null | head -1)

if [ -z "$run_dir" ]; then
    echo "[${EXPERIMENT} group ${GROUP_IDX}] ERROR: Could not find run directory, skipping evaluation."
    exit 1
fi

echo "[${EXPERIMENT} group ${GROUP_IDX}] Evaluating ${run_dir}..."
python src/evaluate.py "$run_dir"
echo "[${EXPERIMENT} group ${GROUP_IDX}] Done."

