#!/bin/bash

source /cvmfs/sft.cern.ch/lcg/views/LCG_105a_cuda/x86_64-el9-gcc11-opt/setup.sh

# ---- CONFIG ----
INPUT_DIR="/vols/cms/mm1221/geant4sim/simulations/build/Test_EM_11_20/"
CODE_DIR="/vols/cms/mm1221/geant4sim/scripts/validation_new"

FILENAME="$1"
INPUT_FILE="${INPUT_DIR}/${FILENAME}"

if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "ERROR: file not found: ${INPUT_FILE}"
  exit 1
fi

export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

mkdir -p dfs

OUTCSV="dfs/${FILENAME%.root}.csv"

python -m scripts.DF_maker \
  -i "${INPUT_FILE}" \
  --out "${OUTCSV}" \
  -clustering_algorithm oc \
  -distance_threshold 0.25 \
  -metric cosine \
  -linkage average \
  -cluster_events 100 \
  -max_events 100

echo "Wrote ${OUTCSV}"

