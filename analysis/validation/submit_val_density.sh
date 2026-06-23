#!/bin/bash
#SBATCH --job-name=hgcal_val_den
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/users/jjchun/scratch/hgcal_cl/logs/val_density_%j.out
#SBATCH --error=/users/jjchun/scratch/hgcal_cl/logs/val_density_%j.err

PYTHON=/users/jjchun/.conda/envs/hgcal/bin/python
VAL_DIR=/oscar/home/jjchun/lgouskos_data/jjchun/hgcal_cl/repo/HGCAL_CL/analysis/validation
DATA_DIR=/oscar/data/lgouskos/jjchun/hgcal_cl/raw/val/raw
MODEL_PATH=/users/jjchun/scratch/hgcal_cl/runs/run01/best_model.pt
OUT_CSV=${VAL_DIR}/dfs/val_density.csv

cd ${VAL_DIR}
mkdir -p dfs

echo "Running DF_maker (density, paper params: t_b=0.4, t_d=0.45)..."
${PYTHON} -u -m scripts.DF_maker \
    -i ${DATA_DIR}/Test_pi.1495_1496.root \
    -o ${OUT_CSV} \
    -task contrastive \
    -cluster density \
    -model_path ${MODEL_PATH} \
    -final_dim 16 \
    -beta_thr 0.4 \
    -oc_td 0.45 \
    -max_events 200

echo "Running metrics_calculate..."
${PYTHON} -u -m scripts.metrics_calculate -i ${OUT_CSV}
