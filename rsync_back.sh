#!/bin/bash


rsync -avz --progress \
  --exclude=".git/" \
  --exclude="*/__pycache__/" \
  jz:/lustre/fswork/projects/rech/ycx/utq99jv/tibo_aimi/SPD_WGFM/runner/logs/ \
  /share/home/okabayas/tibo_aimi/SPD_WGFM/runs_jz/logs/