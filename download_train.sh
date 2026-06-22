#!/bin/bash
#SBATCH --job-name=download_train
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/users/jjchun/scratch/hgcal_cl/logs/download_%j.out
#SBATCH --error=/users/jjchun/scratch/hgcal_cl/logs/download_%j.err

DEST="/oscar/data/lgouskos/jjchun/hgcal_cl/raw/train/raw/Train.root"
URL="https://cernbox.cern.ch/remote.php/dav/public-files/l6rPVXgA9rqBEcm/Train.root"

echo "Starting download at $(date)"
wget --continue --timeout=3600 --tries=5 -O "${DEST}" "${URL}"
STATUS=$?
echo "wget exit code: ${STATUS}"

if [ ${STATUS} -eq 0 ]; then
    SIZE=$(stat -c%s "${DEST}")
    echo "Download complete. File size: ${SIZE} bytes"
else
    echo "Download FAILED."
    rm -f "${DEST}"
fi
echo "Done at $(date)"
