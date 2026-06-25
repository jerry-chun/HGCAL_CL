#!/bin/bash
#SBATCH --job-name=hgcal_val02
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/users/jjchun/scratch/hgcal_cl/logs/val02_%j.out
#SBATCH --error=/users/jjchun/scratch/hgcal_cl/logs/val02_%j.err

CONDA_BASE=/users/jjchun/.conda
PYTHON=${CONDA_BASE}/envs/hgcal/bin/python

VAL_DIR=/oscar/data/lgouskos/jjchun/hgcal_cl/repo/HGCAL_CL/analysis/validation
DATA_DIR=/oscar/data/lgouskos/jjchun/hgcal_cl/raw/val/raw
MODEL_PATH=/users/jjchun/scratch/hgcal_cl/runs/run01/best_model.pt
OUT_DIR=${VAL_DIR}/dfs/val02

cd ${VAL_DIR}
mkdir -p ${OUT_DIR}

VAL_FILES=(
    Test_pi.1495_1496.root
    Test_pi.1869_1870.root
    Test_pi.3445_3446.root
    Test_pi.3681_3682.root
    Test_pi.6311_6312.root
)

echo "=== Running DF_maker on all 5 val files ==="
for f in "${VAL_FILES[@]}"; do
    BASE="${f%.root}"
    echo "Processing ${f}..."
    ${PYTHON} -u -m scripts.DF_maker \
        -i ${DATA_DIR}/${f} \
        -o ${OUT_DIR}/${BASE}.csv \
        -task contrastive \
        -model_path ${MODEL_PATH} \
        -final_dim 16 \
        -distance_threshold 9.5 \
        -max_events 200
done

echo "=== Concatenating CSVs ==="
${PYTHON} -c "
import pandas as pd, glob, os
files = sorted(glob.glob('${OUT_DIR}/*.csv'))
print(f'Combining {len(files)} files...')
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv('${OUT_DIR}/combined.csv', index=False)
print(f'Total rows: {len(df)}')
"

echo "=== Computing metrics on combined CSV ==="
${PYTHON} -u -m scripts.metrics_calculate -i ${OUT_DIR}/combined.csv
