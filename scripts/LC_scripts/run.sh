#!/usr/bin/env bash
set -euo pipefail

INFILE=""
OUTFILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)  INFILE="$2"; shift 2 ;;
    -o|--output) OUTFILE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$INFILE" || -z "$OUTFILE" ]]; then
  echo "Usage: $0 -i <infile> -o <outfile>" >&2
  exit 2
fi

echo "[RUN] $INFILE -> $OUTFILE"
mkdir -p "$(dirname "$OUTFILE")"

WRITER="/vols/cms/mm1221/Independent/LC_scripts/LC_writer.py"
python3 "$WRITER" -i "$INFILE" -o "$OUTFILE"