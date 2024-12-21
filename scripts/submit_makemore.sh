#!/bin/bash
#SBATCH --job-name=makemore
#SBATCH --time=01:00:00
#SBATCH --account=def-jinguo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/breandan/slurm-%j.out
#SBATCH --error=/scratch/breandan/slurm-%j.err
#SBATCH --mail-user=bre@ndan.co
#SBATCH --mail-type=ALL

module load StdEnv/2020
module load python/3.11
source env/bin/activate

pip install --no-index --find-links /cvmfs/soft.computecanada.ca/custom/python/wheelhouse torch tensorboard

python makemore.py