#!/bin/bash
#SBATCH --job-name=hgcal_val
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/users/jjchun/scratch/hgcal_cl/logs/val_%j.out
#SBATCH --error=/users/jjchun/scratch/hgcal_cl/logs/val_%j.err

CONDA_BASE=/users/jjchun/.conda
PYTHON=${CONDA_BASE}/envs/hgcal/bin/python

VAL_DIR=/oscar/home/jjchun/lgouskos_data/jjchun/hgcal_cl/repo/HGCAL_CL/analysis/validation
DATA_DIR=/oscar/data/lgouskos/jjchun/hgcal_cl/raw/val/raw
MODEL_PATH=/users/jjchun/scratch/hgcal_cl/runs/run01/best_model.pt
OUT_CSV=${VAL_DIR}/dfs/val_agglom.csv

cd ${VAL_DIR}
mkdir -p dfs

echo "Running DF_maker..."
${PYTHON} -u -m scripts.DF_maker \
    -i ${DATA_DIR}/Test_pi.1495_1496.root \
    -o ${OUT_CSV} \
    -task contrastive \
    -model_path ${MODEL_PATH} \
    -final_dim 16 \
    -distance_threshold 9.5 \
    -max_events 200

echo "Running metrics_calculate..."
${PYTHON} -u -m scripts.metrics_calculate -i ${OUT_CSV}
