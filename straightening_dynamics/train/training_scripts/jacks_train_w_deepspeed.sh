#!/bin/bash
#SBATCH --job-name=MT_gpt2
#SBATCH --time=2-12:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=evlab

# module load openmind/cuda/11.3

source ~/.bashrc

module load openmind8/cuda/11.7

# find the user name
USER_NAME=$(whoami)
unset CUDA_VISIBLE_DEVICES

SD_HOME="/om2/user/${USER_NAME}/straightening_dynamics/"
# run the .bash_profile file from USER_NAME home directory
# . /home/${USER_NAME}/.bash_profile

conda activate modular_transformers
echo $(which python)

accelerate launch --config_file "${SD_HOME}/straightening_dynamics/train/configs/deepspeed_config.yaml" "${SD_HOME}/straightening_dynamics/train/train.py"