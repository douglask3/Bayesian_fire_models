#!/bin/bash -l
#SBATCH --mem=20000M
#SBATCH --ntasks=5                     # Number of nodes
#SBATCH --output=outputs/ConFLAME_output_%j.txt         # Output file (%j expands to job ID)
#SBATCH --error=outputs/ConFLAME_error_%j.txt           # Error file (%j expands to job ID)
#SBATCH --time=2:00:00 

conda activate bayesian-fire-models
python run_ConFire.py


