import sys
sys.path.append('.')
sys.path.append('make_input/nrt/')
from cf_ensemble_mean import *

if __name__=="__main__":
    run_for_region("Amazon", 'data/data/driving_data_base/')
    
