#!/usr/bin/env bash

HITS="$1"
LC="$2"


OUT_DIR="/vols/cms/mm1221/Independent/Files_merge/photons_2_merged"

source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh

lc_base="$(basename "$LC")"                 
core="${lc_base%_LC.root}"                  
out="${OUT_DIR}/${core}_full.root"          

mkdir -p "$OUT_DIR"

echo "[MERGE] HITS=$HITS"
echo "[MERGE]   LC=$LC"
echo "[MERGE]  OUT=$out"


# Overwrite if exists
hadd -f "$out" "$HITS" "$LC"
echo "[DONE] $out"

