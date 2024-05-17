#!/bin/bash

#SBATCH --job-name="Train MARS Model"
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem=8GB
#SBATCH --account=education-eemcs-courses-cse3000

# Load modules
module load 2023r1
module load openmpi
module load python/3.9

# Create a virtual environment
python -m venv ~/myenv

# Activate the virtual environment
source ~/myenv/bin/activate

# Install python packages
pip install tensorflow
pip install keras
pip install numpy
pip install matplotlib
pip install scikit-learn

srun python MARS_model.py > pi.log
