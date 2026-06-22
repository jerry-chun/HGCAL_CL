#!/bin/bash
#SBATCH --job-name=hgcal_cml
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/users/jjchun/scratch/hgcal_cl/logs/train_%j.out
#SBATCH --error=/users/jjchun/scratch/hgcal_cl/logs/train_%j.err

bash run.sh
