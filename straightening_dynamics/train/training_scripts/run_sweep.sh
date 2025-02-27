#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=jackking@mit.edu
###SBATCH --partition=evlab

source ~/.bashrc

# Load CUDA
module load openmind8/cuda/12.1

# Find the user name
USER_NAME=$(whoami)
unset CUDA_VISIBLE_DEVICES

SD_HOME="/om2/user/${USER_NAME}/straightening_dynamics/"

# Activate conda environment
conda activate modular_transformers
echo $(which python)

# Parse command line arguments
BASE_CONFIG_PATH="${1:-${SD_HOME}/straightening_dynamics/train/configs/train_config.yaml}"
SWEEP_CONFIG_PATH="${2:-${SD_HOME}/straightening_dynamics/train/configs/sweep_config.yaml}"

# Run sweep
python "${SD_HOME}/straightening_dynamics/train/run_sweep.py" --base_config "$BASE_CONFIG_PATH" --sweep_config "$SWEEP_CONFIG_PATH" 