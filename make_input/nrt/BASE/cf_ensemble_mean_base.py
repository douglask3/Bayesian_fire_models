import sys
sys.path.append('.')
sys.path.append('make_input/nrt/')
from cf_ensemble_mean import *

if __name__=="__main__":
    for region in ["Amazon", "Pantanal"]:
        run_for_region(region, 'data/data/driving_data_base/')
    
