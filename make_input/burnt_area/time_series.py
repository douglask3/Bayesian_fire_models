import iris
import iris.analysis
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import pymc.math as pmm
import patsy
import datetime
import cftime

import sys
sys.path.append('../../libs/')
sys.path.append('libs/')
from constrain_cubes_standard import *

from pdb import set_trace

plot_running_mean = True
active_fires = False
country = None
FA_file = "data/data/driving_data2425/burnt_area0p05-uk.nc"
BA_file = "/data/users/douglas.kelley/fireMIPbenchmarking/data/benchmarkData/ISIMIP3a_obs/GFED5_Burned_Percentage.nc"
#FA_file = "/data/users/douglas.kelley/fireMIPbenchmarking/data/benchmarkData/ISIMIP3a_obs/GFED500m_Burned_Percentage.nc"
FA_file = BA_file
#BA_file = "data/data/driving_data2425/burnt_area0p05-uk.nc"
def plot_metric(file_in, file_fig, active_fires = False, analysis_fun = iris.analysis.SUM,
                country = None, spline = False):
    cube = iris.load_cube(file_in)
    cube = sub_year_range(cube, [2002, 2025])
    cube = constrain_natural_earth(cube, 'United Kingdom')

    if active_fires: 
        cube.data = cube.data > 0.0
        ylabel = "Active Fires (no)"
        title = 'UK wide Active Fire Count'
    else:
        ylabel = 'Burned Area (%)'
        title = 'UK wide Burned Area'

        for coord in ['latitude', 'longitude']:
            try:
                cube.coord(coord).guess_bounds()
            except:
                pass
        area_m2 = iris.analysis.cartography.area_weights(cube)
        cube.data *= area_m2 / 4000000.0

    if country is not None:
        cube = constrain_natural_earth(cube, 'England', shape_cat = 'admin_0_map_units')

    # Compute a 12-month running mean
    running_mean_cube = cube.rolling_window('time', iris.analysis.SUM, 12)
    
    # Compute global mean burnt area over time
    global_mean_burnt_area = running_mean_cube.collapsed(['latitude', 'longitude'], 
                                                         analysis_fun).data
    
    global_mean_burnt_area_no_running = cube.collapsed(['latitude', 'longitude'], 
                                                       analysis_fun).data

    # Extract time values and convert to datetime
    time_coord = running_mean_cube.coord('time')
    time_units = time_coord.units
    time_values = [
        datetime.datetime(t.year, t.month, t.day) if isinstance(t, cftime.DatetimeGregorian) else t 
        for t in time_units.num2date(time_coord.points)
    ]
    
    time_numeric = np.arange(len(global_mean_burnt_area))  # Numeric time axis for regression
    if spline:
        # Define the number of knots for the spline
        num_knots = 2  
        knot_positions = np.linspace(time_numeric.min(), time_numeric.max(), num_knots)

        # Create spline basis using patsy
        B = patsy.dmatrix(f"bs(time_numeric, knots={knot_positions[1:-1].tolist()}, degree=3, include_intercept=True)", 
                  {"time_numeric": time_numeric}, return_type="dataframe").values
        
        with pm.Model() as model:
            # Priors for spline coefficients
            beta_spline = pm.Normal("beta_spline", mu=0, sigma=10, shape=B.shape[1])
        
            # Additive model: Sum of spline basis functions weighted by beta coefficients
            mu = pm.Deterministic("mu", pm.math.dot(B, beta_spline))

            # Noise term
            sigma = pm.HalfNormal("sigma", sigma=1)

            # Observed log burnt area
            obs = pm.Normal("obs", mu=mu, sigma=sigma, 
                            observed=np.log(global_mean_burnt_area + 0.001))

            # Sample posterior
            trace = pm.sample(2000, tune=1000, return_inferencedata=True)

        # Compute Bayesian trend line and 90% credible interval
        trend_median = np.exp(np.median(trace.posterior["mu"].values, axis=(0, 1)))
        trend_5 = np.exp(np.percentile(trace.posterior["mu"].values, 5, axis=(0, 1)))
        trend_95 = np.exp(np.percentile(trace.posterior["mu"].values, 95, axis=(0, 1)))
    else:
        # Bayesian Linear Regression (on original scale, not global_mean_burnt_arealog)
        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            beta = pm.Normal("beta", mu=0, sigma=10)
            sigma = pm.HalfNormal("sigma", sigma=1)
            
            mu = alpha + beta * time_numeric
            obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=np.log(global_mean_burnt_area + 0.001))
            trace = pm.sample(2000, tune=1000, return_inferencedata=True)
            
        # Extract posterior samples
        alpha_samples = trace.posterior["alpha"].values.flatten()
        beta_samples = trace.posterior["beta"].values.flatten()
        # Compute the Bayesian trend line and 90% credible interval
        trend_median = np.exp(np.median(alpha_samples) + np.median(beta_samples) * time_numeric)
        trend_5 = np.exp(np.percentile(alpha_samples[:, None] + beta_samples[:, None] * time_numeric, 5, axis=0))
        trend_95 = np.exp(np.percentile(alpha_samples[:, None] + beta_samples[:, None] * time_numeric, 95, axis=0))

    # Plot the time series as a bar plot with Bayesian trend
    #plt.figure(figsize=(10, 5))
    plt.figure(figsize=(6.7, 4))
    plt.style.use("seaborn-v0_8-darkgrid")
    if plot_running_mean:
        #global_mean_burnt_area = global_mean_burnt_area
        bar_X = time_values[0:len(time_values):12]
        bar_Y = global_mean_burnt_area[0:len(global_mean_burnt_area):12]
        width = 200
    else:
        bar_X = time_values
        bar_Y = global_mean_burnt_area_no_running[11:]
        width = 20
        
    plt.bar(bar_X, bar_Y, width=width, label="Monthly Burned Area", 
            color="blue", alpha=0.6)
    plt.plot(time_values, trend_median, label="Trend (%/year)", color="red", linewidth=2)
    plt.fill_between(time_values, trend_5, trend_95, color="red", alpha=0.3, 
                     label="90% Credible Interval")
    
    # Formatting
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.legend()
    plt.savefig("figs/" + file_fig + "_UK_TS.png", format="png", bbox_inches="tight")
    #plt.show()

plot_metric(BA_file, "Burned_area")
plot_metric(BA_file, "Active_fire", True, iris.analysis.SUM)
