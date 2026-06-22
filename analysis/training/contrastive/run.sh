#!/bin/bash

source /oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/etc/profile.d/conda.sh
conda activate hgcal
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main.py
