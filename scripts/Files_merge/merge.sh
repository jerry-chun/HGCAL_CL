#!/usr/bin/env bash

HITS="$1"
LC="$2"

# Where to write outputs
OUT_DIR="/vols/cms/mm1221/Independent/Files_merge/photons_2_merged"

# ROOT from LCG (for `hadd`)
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh

# Derive output name from the LC file: drop "_LC.root" → add "_full.root"
lc_base="$(basename "$LC")"                 
core="${lc_base%_LC.root}"                  
out="${OUT_DIR}/${core}_full.root"          

mkdir -p "$OUT_DIR"

echo "[MERGE] HITS=$HITS"
echo "[MERGE]   LC=$LC"
echo "[MERGE]  OUT=$out"

if [[ ! -f "$HITS" ]]; then
  echo "[WARN] Missing hits file, skipping: $HITS" >&2
  exit 0
fi
if [[ ! -f "$LC" ]]; then
  echo "[WARN] Missing LC file, skipping: $LC" >&2
  exit 0
fi

# Overwrite if exists
hadd -f "$out" "$HITS" "$LC"
echo "[DONE] $out"

