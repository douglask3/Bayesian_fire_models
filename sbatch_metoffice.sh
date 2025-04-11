#!/bin/bash -l
#SBATCH --mem=100M
#SBATCH --ntasks=5                     # Number of nodes
#SBATCH --output=outputs/output_%j.txt         # Output file (%j expands to job ID)
#SBATCH --error=outputs/error_%j.txt           # Error file (%j expands to job ID)
#SBATCH --time=10 

conda activate bayesian-fire-models
python run_ConFire.py


