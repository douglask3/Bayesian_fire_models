#!/bin/bash -l
#SBATCH --mem=200000M
#SBATCH --ntasks=1                     # Number of nodes
#SBATCH --output=outputs/ConFLAME_output_%j.txt         # Output file (%j expands to job ID)
#SBATCH --error=outputs/ConFLAME_error_%j.txt           # Error file (%j expands to job ID)
#SBATCH --time=24:00:00  
#SBATCH --partition=cpu-long


python plot_trend_maps.py
#python make_input/isimip/sow2526.py
#python make_input/isimip/global.py
#python make_input/nrt/SoW2526.py
# Rscript plotting/driver_control_vs_fire.r
#Rscript make_input/isimip/regrid_vcf.r

#python make_input/nrt/burned_area.py
#python make_input/nrt/make_er5_extra_vars.py
#python make_input/nrt/make_factual_counter.py
#python make_input/nrt/make_ConFLAME_inputs.py
#lftp -e "mirror --only-newer SOW_FORCINGS /data/users/douglas.kelley/Bayesian_fire_models/Joeys/; quit" -u ecmwf_fire,FhXekWMuy ftp.ecmwf.int
#cp -r /data/exab/users/eleanor.burke/isimip3b/InputData/climate/atmosphere/historical /data/scratch/douglas.kelley/isimip3/isimip3b/InputData/climate/atmosphere/
