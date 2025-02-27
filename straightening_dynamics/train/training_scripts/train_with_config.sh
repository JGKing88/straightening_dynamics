#!/bin/bash
#SBATCH --job-name=train_model
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
CONFIG_PATH="${1:-${SD_HOME}/straightening_dynamics/train/configs/train_config.yaml}"

# Run training with accelerate
accelerate launch --config_file "${SD_HOME}/straightening_dynamics/train/configs/deepspeed_config.yaml" "${SD_HOME}/straightening_dynamics/train/train.py" --config "$CONFIG_PATH" 