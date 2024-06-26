#!/bin/bash -l
#SBATCH --qos=long  #high #long
#SBATCH --mem=100000  # 100000
#SBATCH --ntasks-per-node=1            # Number of tasks per node
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --output=outputs/output_%j.txt         # Output file (%j expands to job ID)
#SBATCH --error=outputs/error_%j.txt           # Error file (%j expands to job ID)
#SBATCH --time=24:00:00 

conda activate pymc5_env
python run_ConFire.py
