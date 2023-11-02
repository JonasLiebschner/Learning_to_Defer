#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=sdil
#SBATCH --output=slurm_cifar100.%j.out

echo "$PWD"

cd ..

conda activate ./joli-env

cd Masterarbeit

WSDIR_JOLI=$(ws_find liebschner)

srun python -u ./experiment.py $WSDIR_JOLI 0 CIFAR100_Experiment.json