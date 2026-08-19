import sys
sys.path.append('../../libs/')
sys.path.append('libs/')
from constrain_cubes_standard import *

from shapely.ops import unary_union
import shapely.vectorized

import iris
import iris.analysis

import cftime
import cf_units
import numpy as np

import pymc as pm
import pymc_bart as pmb
import arviz as az

from pdb import set_trace


dir = 'data/data/driving_data_base/Amazon/nrt/era5_monthly/'
target_file = 'LI_2003_2025.nc'

explanitory_variables = ['precip.nc', 'max_consec_dry.nc']

target = iris.load_cube(dir + target_file)
target = sub_year_range(target, [2004, 2024])

def load_regrid(file):
    cube = iris.load_cube(dir + file)
    cube = cube.regrid(target, iris.analysis.Linear())  
    cube = sub_year_range(cube, [2004, 2024])
    return cube.data.flatten()

X = np.array([load_regrid(file) for file in explanitory_variables])
Y = target.data.flatten()
#set_trace()
X = X[:,~Y.mask].T
Y = Y[~Y.mask].compressed()

npoints = 100

rng = np.random.default_rng()

samples = rng.choice(len(X), size=npoints, replace=False)
X_train = X[samples,:]
Y_train = Y[samples]

'''
# X is (M, N), y is (M,)
with pm.Model() as model:
    # 1. Define the BART component for the mean
    # m=50 is the number of trees (default)
    mu = pmb.BART("mu", X, Y, m=50)
    
    # 2. Define the observation noise
    sigma = pm.HalfNormal("sigma", sigma=Y.std())
    
    # 3. Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=Y)
    
    # 4. Sample
    idata = pm.sample(cores=1, chains=4)
'''


def create_fire_model(X_data, y_data):
    with pm.Model() as model:
        # MutableData is the "container" that allows swapping
        X_shared = pm.MutableData("X_shared", X_data)
        
        # BART component
        mu = pmb.BART("mu", X_shared, y_data, m=50)
        
        # Priors for observation noise
        sigma = pm.HalfNormal("sigma", sigma=y_data.std() if y_data is not None else 1.0)
        
        # Likelihood
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)
        
    return model

# 1. Create and Sample
model = create_fire_model(X_train, Y_train)
with model:
    idata = pm.sample(cores=1)

az.to_netcdf(idata, "my_bart_model.nc")
#set_trace()

# 2. Save
idata.to_netcdf("fire_model.nc")

# 1. Reload samples
idata_reloaded = az.from_netcdf("fire_model.nc")

# 2. Re-create the EXACT same model structure
# Note: We pass X_train just to initialize dimensions correctly
model_reloaded = create_fire_model(X_train, Y_train)

# 3. Swap in the new data and predict
with model_reloaded:
    pm.set_data({"X_shared": X})
    # Use the samples from the file to generate new 'mu' values
    ppc = pm.sample_posterior_predictive(
        idata_reloaded, 
        var_names=["mu"], 
        predictions=True
    )
    
oos_preds = ppc.predictions["mu"]
mean_prediction = oos_preds.mean(dim=["chain", "draw"])

# 4. Get results
#mean_prediction = ppc.posterior_predictive#["mu"].mean(dim=["chain", "draw"])
set_trace()


