#!/bin/bash -l
#SBATCH --mem=400000M
#SBATCH --ntasks=20                     # Number of nodes
#SBATCH --output=outputs/ConFLAME_output_%j.txt         # Output file (%j expands to job ID)
#SBATCH --error=outputs/ConFLAME_error_%j.txt           # Error file (%j expands to job ID)
#SBATCH --time=6:00:00 

# Get the namelist argument
NAMELIST=$1

conda activate bayesian-fire-models
python run_ConFLAME.py  "$NAMELIST"


