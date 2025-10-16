#!/bin/bash 
# ============================== # Configuration # ============================== 

source /home/hep/mm1221/miniforge3/etc/profile.d/conda.sh
conda activate gnn-cu118



# Call the HyperParam.py script with the passed hyperparameters 
python train.py --model edgeconv --loss contrastive_pairs --temperature 0.1 \
  --batch-size 96 --epochs 10 --run-dir runs/EdgeConv/ --max-events 80000 \
  --train-path "/vols/cms/mm1221/Independent/Data/photons_2/train/" \
  --val-path "/vols/cms/mm1221/Independent/Data/photons_2/val/"
