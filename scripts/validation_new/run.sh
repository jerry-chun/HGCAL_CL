#!/bin/bash

source /cvmfs/sft.cern.ch/lcg/views/LCG_105a_cuda/x86_64-el9-gcc11-opt/setup.sh

BASE_DIR="/vols/cms/mm1221/geant4sim/simulations/build/test_2_10"
RAW_DIR="${BASE_DIR}/raw"

FILENAME="$1"                      # e.g. file_042.root
REALFILE="${RAW_DIR}/${FILENAME}"

if [[ ! -f "${REALFILE}" ]]; then
  echo "ERROR: input file not found: ${REALFILE}"
  exit 2
fi

# Unique per-job workspace folder
JOBTAG="${CLUSTER:-local}.${PROCESS:-0}"
WORKDIR="work_${JOBTAG}"

# Ensure cleanup even if job fails
cleanup() {
  rm -rf "${WORKDIR}"
}
trap cleanup EXIT

mkdir -p "${WORKDIR}/raw" dfs

# Put only this file into the job's raw/ via symlink (fast + no copy)
cp -p "${REALFILE}" "${WORKDIR}/raw/${FILENAME}"
# Output path named after input file
OUTCSV="/vols/cms/mm1221/geant4sim/scripts/validation_new/dfs/${FILENAME%.root}.csv"
PROJECT_DIR="/vols/cms/mm1221/geant4sim/scripts/validation_new/"   # <-- CHANGE THIS

cd "${PROJECT_DIR}"
python -m scripts.DF_maker \
  -i "${WORKDIR}" \
  --out "${OUTCSV}" \
  -clustering_algorithm agglomerative \
  -distance_threshold 0.25 \
  -metric cosine \
  -linkage average \
  -cluster_events 1000 \
  -max_events 1000

echo "Wrote ${OUTCSV}"
ls -lh dfs

