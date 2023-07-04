#!/bin/bash -l
#SBATCH --qos=long  #high #long
#SBATCH --mem=50000  # 50000
#SBATCH --ntasks-per-node=4            # Number of tasks per node
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --output=outputs/output_%j.txt         # Output file (%j expands to job ID)
#SBATCH --error=outputs/error_%j.txt           # Error file (%j expands to job ID)
#SBATCH --time=25:00:00 

conda activate pymc5_env
python pymc_MaxEnt.py
