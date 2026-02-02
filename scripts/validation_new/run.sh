#!/bin/bash

source /cvmfs/sft.cern.ch/lcg/views/LCG_105a_cuda/x86_64-el9-gcc11-opt/setup.sh

INPUT_DIR="/vols/cms/mm1221/geant4sim/simulations/build/Datasets/Test_EM_11_20/"
CODE_DIR="/vols/cms/mm1221/geant4sim/scripts/validation_new"
MODEL_PATH = "/vols/cms/mm1221/geant4sim/scripts/training/ObjectCondensation/runs/EM_2_10_CD3_delta5_1/best_model.pt"
FILENAME="$1"
INPUT_FILE="${INPUT_DIR}/${FILENAME}"

if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "ERROR: file not found: ${INPUT_FILE}"
  exit 1
fi

if [[ ! -f "${MODEL_PATH}"]]; then
  echo "ERROR: Model not found"
  exit 1
fi

export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

mkdir -p dfs

OUTCSV="dfs/${FILENAME%.root}.csv"

python -m scripts.DF_maker \
  -i "${INPUT_FILE}" \
  -o "${OUTCSV}" \
  -task oc \
  -model_path "${MODEL_PATH}" \
  -final_dim 16 \
  -distance_threshold 0.25 \
  -oc_td 0.3 \
  -max_events 100
  

echo "Wrote ${OUTCSV}"

