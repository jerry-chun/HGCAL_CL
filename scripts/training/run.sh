#!/bin/bash 
# ============================== # Configuration # ============================== 

source /home/hep/mm1221/miniforge3/etc/profile.d/conda.sh
conda activate gnn-cu118


python train.py --model edgeconv --loss contrastive_pairs --temperature 0.01 \
  --batch-size 96 --epochs 30 --run-dir runs/edgeconv --max-events 650000 \
  --train-path "/vols/cms/mm1221/Independent/Data/photons_2/train/" \
  --val-path "/vols/cms/mm1221/Independent/Data/photons_2/val/"
