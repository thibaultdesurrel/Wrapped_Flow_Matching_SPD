#!/bin/bash

rsync -avz --progress \
  --exclude=".git/" \
  --exclude="*/__pycache__/" \
  --exclude="*/logs/" \
  --exclude="*/runs/" \
  --exclude=".venv/" \
  --exclude="job_logs/" \
  --exclude="runs_jz" \
  --exclude="results/" \
  --exclude="venvspd/" \
  --exclude="notebooks" \
  --exclude="datasets/breizhcrops/2017" \
  /share/home/okabayas/tibo_aimi/SPD_WFM_2/ jz:/lustre/fswork/projects/rech/ycx/utq99jv/tibo_aimi/SPD_WFM_2