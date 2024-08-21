
from train import *
from evaluate import *
from run_ConFire import *

from   io     import StringIO
import numpy  as np
import cftime

import matplotlib.pyplot as plt 

def run_ConFire_nrt(namelist):   
    
    run_info = read_variables_from_namelist(namelist) 

    regions = run_info['regions']
    subset_function_args = run_info['subset_function_args'] 

    for region in regions:
        model_title = run_info['model_title'].replace('<<region>>', region)
        dir_training = run_info['dir_training'].replace('<<region>>', region)

        trace, scalers, training_namelist = \
                        train_MaxEnt_model_from_namelist(namelist, model_title = model_title,
                                                         dir_training = dir_training,
                                                         subset_function_args = subset_function_args)
        
        params = read_variables_from_namelist(training_namelist)
        output_dir = params['dir_outputs']
        output_file = params['filename_out']
        
        control_direction = read_variables_from_namelist(params['other_params_file'])
        control_direction = control_direction['control_Direction']
        try:
            control_names = read_variables_from_namelist(namelist)['control_names']
        except:
            control_names = None
        
        subset_function_args = read_variables_from_namelist(namelist)['subset_function_args_eval']
        
        run_experiment(training_namelist, namelist, control_direction, control_names,
                                  output_dir, output_file, 'baseline', 
                                  model_title = model_title, 
                                  subset_function_args = subset_function_args)


if __name__=="__main__":
    namelist = 'namelists/nrt.txt'
    run_ConFire_nrt(namelist)
    namelist = 'namelists/nrt-evaluation.txt'
    run_ConFire_nrt(namelist)
    namelist = 'namelists/isimip-evaluation.txt'
    run_ConFire_nrt(namelist)

