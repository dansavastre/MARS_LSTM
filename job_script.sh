#!/bin/bash

#SBATCH --job-name="train_MARS"
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --account=Education-EEMCS-Courses-CSE3000

# Load necessary modules
module load python

# Create a Virtual Environment and install dependencies
python -m venv myenv
python -m pip install tensorflow==2.2.0 keras==2.3.0 matplotlib scikit-learn protobuf==3.20
./myenv/Scripts/activate.bat

# Run the MARS model
python ./MARS_LSTM/MARS_model.py

