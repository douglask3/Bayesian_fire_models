'''import numpy as np

class DummyModel:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        # Compute some random coefficients
        n_features = X.shape[1]
        self.coefficients = np.random.randn(n_features)

    def predict(self, X):
        # Compute the dot product of the coefficients and the input features
        return np.dot(X, self.coefficients)

import numpy as np
from sklearn.linear_model import LinearRegression
from pdb import set_trace
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(0)
n_samples = 100
n_features = 5

x = np.random.randn(n_samples, n_features)
x_o = x[:, 0].copy()  # Explanatory variables

#set_trace()
#coefficients = np.random.randn(n_features)  # coefficients
y = np.random.normal(0, 1, n_samples)  # Response variable

# Create a dummy model and fit the data
model = LinearRegression()
model.fit(x, y)
predictions = model.predict(x)

print("Predictions:", predictions)

x[:, 0] = 0

predictions2 = model.predict(x)
print("Predictions2:", predictions2)

new_X = np.random.randn(10, n_features)  # New explanatory variables
new_X[:, 0] = 0
predictions = model.predict(new_X)

print("P2 - P1:", predictions2 - predictions)


plt.plot(x_o, (predictions2/predictions), '.')
plt.show()

#print("Predictions:", predictions)
#set_trace()

#calculate the difference for all and plot all the variables in one big plot side by side'''


import sys
sys.path.append('fire_model/')
sys.path.append('libs/')

from MaxEntFire import MaxEntFire

from read_variable_from_netcdf import *
from plot_maps import *
import os
from   io     import StringIO
import numpy  as np
import math

import pymc  as pm
import pytensor
import pytensor.tensor as tt

import matplotlib.pyplot as plt
import re

import arviz as az


def MaxEnt_on_prob(BA, fx):
    """calculates the log-transformed continuous logit likelihood for x given mu when x 
       and mu are probabilities between 0-1. 
       Works with tensor variables.   
    Arguments:
        x -- x in P(x|mu). tensor 1-d array
	mu -- mu in P(x|mu). tensor 1-d array
    Returns:
        1-d tensor array of liklihoods.
    """
    fx = tt.switch(
        tt.lt(fx, 0.0000000000000000001),
        0.0000000000000000001, fx)
    return BA*tt.log(fx) + (1.0-BA)*tt.log((1-fx))   


def fit_MaxEnt_probs_to_data(Y, X, niterations, 
                             out_dir = 'outputs/', filename = '', grab_old_trace = True,
                             *arg, **kw):
    """ Bayesian inerence routine that fits independant variables, X, to dependant, Y.
        Based on the MaxEnt solution of probabilities. 
    Arguments:
        Y-- dependant variable as numpy 1d array
	X -- numpy 2d array of indepenant variables, each columne a different variable
	niterations -- number of iterations per chain when sampling the postior during 
                NUTS inference 
		(note default chains is normally 2 and is set by *args or **kw)
	out_dir --string of path to output location. This is where the traces netcdf file 
                will be saved.
		Defauls is 'outputs'.
	filename -- string of the start of the traces output name. Detault is blank. 
		Some metadata will be saved in the filename, so even blank will 
                save a file.
	grab_old_trace -- Boolean. If True, and a filename starting with 'filename' and 
                containing some of the same setting (saved in filename) exists,  it will open 
                and return this rather than run a new one. Not all settings are saved for 
                identifiation, so if in doubt, set to 'False'.
	*args, **kw -- arguemts passed to 'pymc.sample'

    Returns:
        pymc traces, returned and saved to [out_dir]/[filneame]-[metadata].nc
    """

    trace_file = out_dir + '/' + filename + '-nvariables_' + '-ncells_' + str(X.shape[0]) + \
                str(X.shape[1]) + '-niterations_' + str(niterations * cores) + '.nc'
    
    ## check if trace file exsts and return if wanted
    if os.path.isfile(trace_file) and grab_old_trace: 
        return az.from_netcdf(trace_file)

    trace_callback = None
    try:
        if "SLURM_JOB_ID" in os.environ:
            def trace_callback(trace, draw):        
                if len(trace) % 10 == 0:
                    print('chain' + str(draw[0]))
                    print('trace' + str(len(trace)))
    except:
        pass        

    with pm.Model() as max_ent_model:
        ## set priorts
        betas = pm.Normal('betas', mu = 0, sigma = 1, shape = X.shape[1], 
                          initval =np.repeat(0.5, X.shape[1]))

        powers = pm.Normal('powers', mu = 0, sigma = 1, shape = [2, X.shape[1]])
        ## build model
        model = MaxEntFire(betas, powers, inference = True)
        prediction = model.fire_model(X)  
        
        
        ## define error measurement
        error = pm.DensityDist("error", prediction, logp = MaxEnt_on_prob, observed = Y)
                
        ## sample model
        
        trace = pm.sample(niterations, return_inferencedata=True, 
                          callback = trace_callback, *arg, **kw)
        ## save trace file
        trace.to_netcdf(trace_file)
    return trace


def train_MaxEnt_model(y_filen, x_filen_list, dir = '', filename_out = '',
                       dir_outputs = '',
                       frac_random_sample = 1.0,
                       subset_function = None, subset_function_args = None,
                       niterations = 100, cores = 4, grab_old_trace = False):
    
   
    Y, X, lmask, scalers = read_all_data_from_netcdf(y_filen, x_filen_list, 
                                                     add_1s_columne = True, dir = dir,
                                                     x_normalise01 = True, 
                                                     frac_random_sample = frac_random_sample,
                                                     subset_function = subset_function, 
                                                     subset_function_args = subset_function_args)
    
    trace = fit_MaxEnt_probs_to_data(Y, X, out_dir = dir_outputs, filename = filename, 
                                     niterations = niterations, cores = cores,
                                     grab_old_trace = grab_old_trace)
    
    az.plot_trace(trace)
    fig_dir = dir_outputs + '/figs/'
    if not os.path.exists(fig_dir): os.makedirs(fig_dir)

    plt.savefig(fig_dir + filename + '-traces.png')
    
    return trace, scalers

def predict_MaxEnt_model(trace, y_filen, x_filen_list, scalers, dir = '', 
                         dir_outputs = '', filename_out = '',
                         subset_functionm = None, subset_function_args = None,
                         sample_for_plot = 1):
    def select_post_param(name): 
        out = trace.posterior[name].values
        A = out.shape[0]
        B = out.shape[1]
        new_shape = ((A * B), *out.shape[2:])
        return np.reshape(out, new_shape)

    def sample_model(i): 
        powers =select_post_param('powers')[i,:]
        betas =select_post_param('betas')[i,:]
        model = MaxEntFire(betas, powers)
        return model.fire_model(X)
    
    nits = np.prod(trace.posterior['betas'].values.shape[0:2])
    idx = range(0, nits, int(np.floor(nits/sample_for_plot)))
                         
    Obs = read_variable_from_netcdf(y_filen, dir,
                                    subset_function = subset_function, 
                                    subset_function_args = subset_function_args)

    Y, X, lmask, scalers = read_all_data_from_netcdf(y_filen, x_filen_list, 
                                                     add_1s_columne = True, dir = dir,
                                                     x_normalise01 = True, scalers = scalers,
                                                     subset_function = subset_function, 
                                                     subset_function_args = subset_function_args)
                                                     
        
                                                     
    #fig, axes = plt.subplots(3, X.shape[1], figsize=(12, 4))  # Create subplots for each variable
    
    for col in range(X.shape[1]):
        x_copy = X[:, col].copy()  # Copy the values of the current column
        print(col)
        Sim = np.array(list(map(sample_model, idx)))  # Sample model for the current column
        
        X[:, col] = 0  # Set the current column to 0
        
        Sim2 = np.array(list(map(sample_model, idx)))  # Sample model for the modified column
    
        ax = plt.subplot(6,4, col + 1)  # Select the corresponding subplot
        #if col == 20:
            #set_trace()
        for rw in range(Sim.shape[0]):
        
            ax.plot(x_copy, (Sim[rw, :] / Sim2[rw, :]), '.')  # Plot the data for the current variable
        
        X[:, col] = x_copy 
        
    plt.show()
    
    set_trace() 

    def insert_sim_into_cube(x):
        Pred = Obs.copy()
        pred = Pred.data.copy().flatten()
        
        pred[lmask] = x
        Pred.data = pred.reshape(Pred.data.shape)
        return(Pred)

    def plot_map(cube, plot_name, plot_n):
        plot_annual_mean(cube, levels, cmap, plot_name = plot_name, scale = 100*12, 
                     Nrows = 1, Ncols = 3, plot_n = plot_n)
  
    plot_map(Obs, "Observations", 1)
    plot_map(insert_sim_into_cube(Sim[0,:]), "Simulation - 10%", 2)
    plot_map(insert_sim_into_cube(Sim[1,:]), "Simulation - 90%", 3)
    plt.gcf().set_size_inches(8, 6)
    
    fig_dir = dir_outputs + '/figs/'
    if not os.path.exists(fig_dir): os.makedirs(fig_dir)
    plt.savefig(fig_dir + filename_out + '-maps.png')


if __name__=="__main__":
    """ Running optimization and basic analysis. 
    Variables that need setting:
    For Optimization:
        dir_training -- The directory of the training data inclduing 
            dependant and independant variables
        y_filen -- filename of dependant variable (i.e burnt area)
        x_filen_list -- filanames of independant variables
            (ie bioclimate, landscape metrics etc)
        cores - how many chains to start (confusiong name, I know).
            When running on slurm, also number of cores
        fraction_data_for_sample -- fraction of gridcells used for optimization
        niterations -- number of iterations or samples )after warm-up) in optimixation for each
            chain. Equilivent to number of ensemble members.
        months_of_year --- which months to extact on training and projecting
        grab_old_trace -- Boolean. If True and there's an appripritate looking old trace file, 
            then  optimisation is skipped that file is loaded instead. 
            This isn't totally infalable, so if doing a final run and in doubt, set to False
    For Projection/evaluating:
        dir_outputs -- where stuff gets outputted
        dir_projecting -- The directory of the data used for prjections. 
            Should contain same files for independant varibales as dir_training 
            (though you should be able to adpated this easily if required). 
            Doesnt need dependant variable, but if there, this will (once
            we've implmented it) attempt some evaluation.
        sample_for_plot -- how many iterations (samples) from optimixation should be used 
            for plotting and evaluation.
        levels -- levels on teh colourbar on observtation and prodiction maps
        cmap -- levels on teh colourbar on observtation and prodiction maps
    Returns:
        trace file, maps, etc (to be added too)
    """

    """ 
        SETPUT 
    """
    """ optimization """
    #dir_training = "../ConFIRE_attribute/isimip3a/driving_data/GSWP3-W5E5-20yrs/Brazil/AllConFire_2000_2009/"
    #dir_training = "/gws/nopw/j04/jules/mbarbosa/driving_and_obs_overlap/AllConFire_2000_2009/"
    dir_training = "D:/Doutorado/Sanduiche/research/maxent-variables/2002-2011/"
    
    y_filen = "GFED4.1s_Burned_Fraction.nc"
    #y_filen = "Area_burned_NAT.nc"
    #"MPA.nc", "TCA.nc", "E_density.nc"


    x_filen_list=["Forest.nc", "dry_days.nc", "consec_dry_mean.nc", "lightn.nc", 
                  "crop.nc", "Savanna.nc", "Grassland.nc", "N_patches.nc", "MPA.nc",
                  "TCA.nc", "E_density.nc",
                  "humid.nc","vpd.nc", "csoil.nc", "tas.nc", "tas_max.nc",
                  "lightn.nc", "rhumid.nc", "cveg.nc", "pas.nc", "soilM.nc", 
                  "popDens.nc"]
                  

    grab_old_trace = True
    cores = 4
    fraction_data_for_sample = 0.1
    niterations = 100

    months_of_year = [7]
    
    """ Projection/evaluating """
    dir_outputs = 'outputs/'

    #dir_projecting = "../ConFIRE_attribute/isimip3a/driving_data/GSWP3-W5E5-20yrs/Brazil/AllConFire_2010_2019/"
    #dir_projecting = "/gws/nopw/j04/jules/mbarbosa/driving_and_obs_overlap/AllConFire_2010_2019/"
    dir_projecting = "D:/Doutorado/Sanduiche/research/maxent-variables/2012-2021/"

    sample_for_plot = 20

    levels = [0, 0.1, 1, 2, 5, 10, 20, 50, 100] 
    cmap = 'OrRd'

    
    """ 
        RUN optimization 
    """
    subset_function = sub_year_months
    subset_function_args = {'months_of_year': months_of_year}

    filename = '_'.join([file[:-3] for file in x_filen_list]) + \
              '-frac_points_' + str(fraction_data_for_sample) + \
              '-Month_' +  '_'.join([str(mn) for mn in months_of_year])

    #### Optimize
    trace, scalers = train_MaxEnt_model(y_filen, x_filen_list, dir_training, 
                                        filename, dir_outputs,
                                        fraction_data_for_sample,
                                        subset_function, subset_function_args,
                                        niterations, cores, grab_old_trace)


    """ 
        RUN projection 
    """
    predict_MaxEnt_model(trace, y_filen, x_filen_list, scalers, dir_projecting,
                         dir_outputs, filename,
                         subset_function, subset_function_args,
                         sample_for_plot)
    

    '''
    #Run the model with first iteration
    simulation1 = fire_model(trace.posterior['betas'].values[0,0,:], X, False)

    #Plot against observations (Y)
    plt.plot(Y, simulation1, '.')
    plt.show()

    #when developing plots, use betas = trace.posterior['betas'].values[0,0,:]
    and run with model with fire_model(betas, X, False)
    '''
    set_trace()
