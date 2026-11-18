import iris
import pymc as pm
import arviz as az
import numpy as np
import sys
sys.path.append('../../libs/')
sys.path.append('libs/')
from constrain_cubes_standard import *

country = 'United Kingdom'
dataset_file = "data/data/driving_data2425/burnt_area0p05-uk.nc"

cube = iris.load_cube(dataset_file)
cube = constrain_natural_earth(cube, country)

# Compute global mean burnt area over time
global_mean_burnt_area = cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN).data
log_burnt_area = np.log(global_mean_burnt_area + 0.000001)

# Convert time to numeric values
time_numeric = np.arange(len(log_burnt_area))  # Assuming time is evenly spaced

# Bayesian Linear Regression
with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=1)

    mu = alpha + beta * time_numeric
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=log_burnt_area)

    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

# Extract the 90% credible interval for the trend coefficient
beta_samples = trace.posterior["beta"].values.flatten()
beta_90_interval = np.percentile(beta_samples, [5, 95])

print(f"90% credible interval for trend coefficient: {beta_90_interval}")

# Plot the trend with uncertainty
az.plot_posterior(trace, var_names=["beta"])#, credible_interval=0.9)
plt.title("Posterior Distribution of Trend Coefficient (90% CI)")
plt.show()
