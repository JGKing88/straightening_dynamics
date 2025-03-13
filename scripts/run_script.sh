#!/bin/bash
#SBATCH --job-name=sd
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=evlab
#SBATCH --mem=100G

source ~/.bashrc

module load openmind8/cuda/12.1
# find the user name
USER_NAME=$(whoami)
unset CUDA_VISIBLE_DEVICES

SD_HOME="/om2/user/${USER_NAME}/straightening_dynamics/"

conda activate superurop
echo $(which python)

python "${SD_HOME}/scripts/paper_verification.py"