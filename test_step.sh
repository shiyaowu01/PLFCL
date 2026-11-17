#!/bin/bash
#
# Example SLURM job script for ALICE3
# SLURM directives:
#SBATCH --job-name=exp2
#SBATCH --account=handpose
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --export=NONE
#SBATCH --gres=gpu:ampere:1

# commands to run
module purge
module load gcc/12.3.0
#source env-name/bin/activate
source ~/miniforge3/bin/activate plfcl
#cd PycharmProjects/HandOcc_step/main
python train_plfcl.py
#python test.py --gpu 0 --test_epoch 1
#python test.py --gpu 0 --test_epoch 2
#python test.py --gpu 0 --test_epoch 70

# just list loaded modules, host we are running on and date
module list
hostname
date
sleep 60
