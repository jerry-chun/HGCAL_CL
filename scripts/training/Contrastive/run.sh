#!/bin/bash

# ==============================
# Configuration
# ==============================


source /home/hep/mm1221/miniforge3/etc/profile.d/conda.sh
conda activate torch25cuda124
# Call the HyperParam.py script with the passed hyperparameters
python main.py

