#!/bin/bash


rsync -avz --progress \
  --exclude=".git/" \
  --exclude="*/__pycache__/" \
  --exclude="**/checkpoints/" \
  jz:/lustre/fswork/projects/rech/ycx/utq99jv/tibo_aimi/SPD_WFM_2/runner/logs/ \
  /share/home/okabayas/tibo_aimi/SPD_WFM_2/runs_jz/logs/