#!/bin/bash

# ==============================
# Configuration
# ==============================
source /home/hep/mm1221/miniforge3/etc/profile.d/conda.sh
conda activate gnn-cu118

#source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh


# Call the HyperParam.py script with the passed hyperparameters
python main.py

