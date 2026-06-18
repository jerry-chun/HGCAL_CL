#!/bin/bash
set -euo pipefail
LIST="Optimal.txt"
N=$(wc -l < "$LIST")
NJOBS=$(( (N + 7) / 8 ))
echo "List has $N files -> submitting $NJOBS condor jobs (8 files/job)"

condor_submit -queue "${NJOBS}" submit.sub