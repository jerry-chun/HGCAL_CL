#!/bin/bash
#set -euo pipefail

source /cvmfs/sft.cern.ch/lcg/views/LCG_105a_cuda/x86_64-el9-gcc11-opt/setup.sh

INPUT_DIR="/vols/cms/mm1221/geant4sim/simulations/build/Datasets/Test_EM_2_10_delta5"
CODE_DIR="/vols/cms/mm1221/geant4sim/scripts/validation_new"
MODEL_PATH="/vols/cms/mm1221/geant4sim/scripts/training/Contrastive/runs/EM_2_10_CD16_Supcon_delta5_t01/best_model.pt"

LISTFILE="$1"          # e.g. EM_2_10_delta_5.txt
CHUNK_ID="$2"          # $(Process)
NCPUS="${3:-8}"        # should match request_cpus; default 8

if [[ ! -f "${LISTFILE}" ]]; then
  echo "ERROR: list file not found: ${LISTFILE}"
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "ERROR: Model not found: ${MODEL_PATH}"
  exit 1
fi

export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
mkdir -p dfs logs

# Select the lines for this chunk:
# chunk 0 -> lines 1..8
# chunk 1 -> lines 9..16
# etc.
START_LINE=$(( CHUNK_ID * NCPUS + 1 ))
END_LINE=$(( START_LINE + NCPUS - 1 ))

mapfile -t FILES < <(sed -n "${START_LINE},${END_LINE}p" "${LISTFILE}" | sed '/^\s*$/d')

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "[chunk ${CHUNK_ID}] No files in range ${START_LINE}-${END_LINE}. Exiting."
  exit 0
fi

echo "=================================================="
echo "Chunk ID: ${CHUNK_ID}"
echo "NCPUS:    ${NCPUS}"
echo "Lines:    ${START_LINE}-${END_LINE}"
echo "Files:    ${#FILES[@]}"
printf '  - %s\n' "${FILES[@]}"
echo "=================================================="

run_one() {
  local FILENAME="$1"
  local INPUT_FILE="${INPUT_DIR}/${FILENAME}"

  if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "WARN: missing input: ${INPUT_FILE} (skipping)"
    return 0
  fi

  local OUTCSV="dfs/${FILENAME%.root}.csv"

  python -m scripts.DF_maker \
    -i "${INPUT_FILE}" \
    -o "${OUTCSV}" \
    -task contrastive \
    -model_path "${MODEL_PATH}" \
    -final_dim 16 \
    -distance_threshold 0.15 \
    -oc_td 0.25 \
    -max_events 1000

  echo "Wrote ${OUTCSV}"
}

export -f run_one
export INPUT_DIR CODE_DIR MODEL_PATH PYTHONPATH

# Run up to NCPUS in parallel (uses 8 CPUs as requested)
if command -v parallel >/dev/null 2>&1; then
  printf "%s\n" "${FILES[@]}" | parallel -j "${NCPUS}" run_one {}
else
  # portable fallback
  for f in "${FILES[@]}"; do
    while [[ $(jobs -r | wc -l) -ge "${NCPUS}" ]]; do
      sleep 0.2
    done
    run_one "$f" &
  done
  wait
fi
