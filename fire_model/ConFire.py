import numpy as np
import iris
import pandas as pd

import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib.pyplot as plt
from pdb import set_trace

import sys
sys.path.append('libs/')
sys.path.append('../libs/')
from select_key_or_default import *

class ConFire(object):
    def __init__(self, params, inference = False):
        """
        Initalise parameters and calculates the key variables needed to calculate burnt area.
        """
        self.inference = inference
        if self.inference:
            self.numPCK =  __import__('pytensor').tensor
        else:
            self.numPCK =  __import__('numpy')
        
        self.params = params

        def select_param_or_default(*args, **kw):
            return select_key_or_default(self.params, numPCK = self.numPCK, *args, **kw) 

        self.controlID = self.params['controlID']
        self.control_Direction = self.params['control_Direction']
        self.x0 = select_param_or_default('x0', [0])
        self.betas = select_param_or_default('betas', [[0]], stack = False)
        self.powers = select_param_or_default('powers', None, stack = False)
        self.driver_Direction = self.params['driver_Direction']
        self.Fmax = select_param_or_default('Fmax', 1.0, stack = False)

    def burnt_area(self, X, return_controls = False, return_limitations = False):
        ## finds controls        
        def cal_control(cid = 0):
            ids = self.controlID[cid]
            betas =  self.betas[cid] * self.driver_Direction[cid]
        
            X_i = X[:,ids]
            if self.powers is not None:
                X_i = self.numPCK.power(X_i, self.powers[cid])
            return self.numPCK.sum(X[:,ids] * betas[None, ...], axis=-1)
            
        
        controls = [cal_control(i) for i in range(len(self.controlID))]
        if return_controls: return controls

        def sigmoid(y, k):
            if k == 0: return None
            return 1.0/(1.0 + self.numPCK.exp(-y * k))
        
        
        limitations = [sigmoid(y, k) for y, k in zip(controls, self.control_Direction)]
        
        if return_limitations:
            return limitations

        limitations = [lim for lim in limitations if lim is not None]
        
        BA = sigmoid(self.Fmax, 1.0) * self.numPCK.prod(limitations, axis = 0)
        return BA
    
    
    def emc_weighted(self, emc, precip, wd_pg):
        
        try:
            wet_days = 1.0 - self.numPCK.exp(-wd_pg * precip)
            emcw = (1.0 - wet_days) * emc + wet_days
        except:
            emcw = emc.copy()
            emcw.data  = 1.0 - self.numPCK.exp(-wd_pg * precip.data)
            emcw.data = emcw.data + (1.0 - emcw.data) * emc.data
        return(emcw)

    def list_model_params(self, params, varnames = None):
        controlID = params[0]['controlID']
        def list_one_line_of_parmas(param):
            def select_param_or_default(*args, **kw):
                return select_key_or_default(param, numPCK = __import__('numpy'), *args, **kw)
            
            control_Direction = param['control_Direction']
            x0s = select_param_or_default('x0', [0])
            betas = select_param_or_default('betas', [[0]], stack = False)
            powers = select_param_or_default('powers', None, stack = False)
            driver_Direction = param['driver_Direction']
            Fmax = select_param_or_default('Fmax', 1.0, stack = False)


            directions = [np.array(driver)*control for control, driver in \
                          zip(control_Direction, driver_Direction)]
            betas = [direction * beta for direction, beta in zip(directions , betas)]
            

            def mish_mash(beta, power, x0):
                return np.append(np.column_stack((beta, power)).ravel(), x0)
            varps = [mish_mash(beta, power, x0) for beta, power, x0 in zip(betas, powers, x0s)]
            
            return np.append(Fmax, np.concatenate(varps))
            
        params_sorted = np.array([list_one_line_of_parmas(param) for param in params])
    
        param = ['Fmax']
        controlN = ['']
        variable_name = ['']
        
        for IDs, IDn in zip(controlID, range(len(controlID))):
            for ID in IDs:
                param.append('beta')
                controlN.append(str(IDn))
                variable_name.append(varnames[ID])
                param.append('power')
                controlN.append(str(IDn))
                variable_name.append(varnames[ID])
            param.append('beta')
            controlN.append(str(IDn))
            variable_name.append('beta0')

        header_df = pd.DataFrame([param, controlN, variable_name])   
        index_labels = ["Parameter", "Control", "Variable"] + \
                        list(map(str, range(1, params_sorted.shape[0] + 1)))

        data_df = pd.DataFrame(params_sorted)
        full_df = pd.concat([header_df, data_df], ignore_index=True) 

        full_df.index = index_labels

        return full_df


if __name__=="__main__":


    from pymc_extras import *
    from namelist_functions import *
    from read_variable_from_netcdf import *
    from iris_plus import insert_data_into_cube
    import iris
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import pandas as pd
    import numpy as np
    import os

    # --------------------------------------------------------
    # **Pantanal Example: Running ConFire Model**
    # This script runs the ConFire model using pre-generated parameter files.
    # It calculates burnt area under different control conditions and saves results.
    # --------------------------------------------------------

    # **Define Paths for Parameter Files  and for outputs**
    info_dir = "outputs/outputs/ConFire_Pantanal_example/"
    output_dir = "outputs/outputs/ConFire_Pantanal_example/simple_output_example"

    # Parameter files (traces, scalers, and other model parameters)
    param_file_trace = info_dir + "trace-_6-frac_points_0.8-nvariables_-frac_random_sample0.8-nvars_6-niterations_4000.nc"
    param_file_none_trace = info_dir + "none_trace-params-_6-frac_points_0.8-nvariables_-frac_random_sample0.8-nvars_6-niterations_4000.txt"
    scale_file = info_dir + "scalers-_6-frac_points_0.8-nvariables_-frac_random_sample0.8-nvars_6-niterations_4000.csv"

    # Number of samples to run from the trace file
    nsample_for_running = 100

    # **Load Variable Information and NetCDF Files**
    # Replace these lines with user-specified NetCDF files, ensuring variable order is the same.
    variable_file = info_dir + "variables_info-_6-frac_points_0.1-nvariables_-frac_random_sample0.1-nvars_6-niterations_4000.txt"
    nc_files = read_variables_from_namelist(variable_file)['x_filen_list']
    nc_dir = read_variables_from_namelist(variable_file)['dir']

    # **Load Driving Data and Land Mask**
    scalers = pd.read_csv(scale_file).values
    obs_data, driving_data, lmask, scalers = read_all_data_from_netcdf(nc_files[0], nc_files, scalers=scalers, dir=nc_dir)

    # Load a sample cube (used for inserting data)
    eg_cube = read_variable_from_netcdf(nc_files[0], dir=nc_dir)

    # **Extract Model Parameters**
    params, params_names = select_post_param(param_file_trace)
    extra_params = read_variables_from_namelist(param_file_none_trace)
    control_direction = extra_params['control_Direction'].copy()
    Nexp = len(control_direction)

    # **Sample Iterations from Trace**
    nits = len(params[0])
    idx = range(0, nits, int(np.floor(nits/nsample_for_running)))

    # **Storage for Model Outputs (Including full model, physical & stochastic controls)**
    out_cubes = [[] for _ in range(Nexp + 2)]

    def run_model_into_cube(param_in, coord):
        """
        Runs the ConFire model with given parameters and inserts results into an Iris cube.
        """
        out = ConFire(param_in).burnt_area(driving_data)
        cube = insert_data_into_cube(out, eg_cube, lmask)
        cube.add_aux_coord(coord)
        return cube

    # **Run ConFire Model with Different Control Scenarios**
    for id, i in zip(idx, range(len(idx))):
        coord = iris.coords.DimCoord(i, "realization")
        param_in = contruct_param_comb(id, params, params_names, extra_params)
        
        Fmax = param_in['Fmax0'].copy()
        # **Run Full Model**
        out_cubes[0].append(run_model_into_cube(param_in, coord))

        # **Run Model with Individual Controls Turned On**
        for cn in range(Nexp):
            param_in['control_Direction'][:] = [0] * Nexp
            param_in['control_Direction'][cn] = control_direction[cn]
            param_in = contruct_param_comb(cn, params, params_names, extra_params)
            param_in['Fmax0'] = np.array([1.0])
            out_cubes[cn+1].append(run_model_into_cube(param_in, coord))

        # **Run Model with All Controls Off (Stochastic Control)**
        param_in['control_Direction'][:] = [0] * Nexp
        param_in['Fmax0'] = Fmax
        out_cubes[cn+2].append(run_model_into_cube(param_in, coord))


    # **Save Output Cubes**
    filenames_out = ['model'] + ['control_' + str(i) for i in range(Nexp)] + \
                    ['control_stochastic']
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(out_cubes)):
        cubes = iris.cube.CubeList(out_cubes[i]).merge_cube()
        iris.save(cubes, os.path.join(output_dir, filenames_out[i] + '.nc'))

    # --------------------------------------------------------
    # **Visualization: Plot Resultant Maps**
    # --------------------------------------------------------
    # **Load and Plot Output Data**
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()   # Flatten to handle looping easily
    plotN = 0
    for i, filename in enumerate(filenames_out):
        filepath = os.path.join(output_dir, filename + ".nc")
        
        try:
            cube = iris.load_cube(filepath)   # Load saved NetCDF file
            annual_avg = cube.collapsed('time', iris.analysis.MEAN)
            for pc in [5, 95]:
                pc_cube = cube.collapsed('realization', iris.analysis.PERCENTILE, percent=pc)[0]
                plotN = plotN + 1
                ax = axes[plotN]
                img = iris.quickplot.pcolormesh(pc_cube, axes=ax)
                ax.set_title(filename.replace("_", " ").capitalize() +'-' + str(pc) + '%')
                ax.coastlines()
                # Add colorbar
                fig.colorbar(img, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")

    plt.tight_layout()
    try:
        plt.savefig(os.path.join(output_dir, "Confire_Results.png"), dpi=300)
    except:
        pass
    plt.show()
    





