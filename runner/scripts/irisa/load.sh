#!/bin/bash -l
#SBATCH --chdir=/share/home/okabayas/tibo_aimi/SPD_WFM_2/
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --partition=shortrun
#SBATCH --output=runs_jz/job_logs/load_breizhcrops_%a.out
#SBATCH --error=runs_jz/job_logs/load_breizhcrops_%a.err
#SBATCH --time=1:00:00

set -e

setcuda 12.1
source venvspd/bin/activate

SEEDS=(0 1 2 3 4)

python runs_jz/cache_breizhcrops.py --seed "${SEEDS[$SLURM_ARRAY_TASK_ID]}"