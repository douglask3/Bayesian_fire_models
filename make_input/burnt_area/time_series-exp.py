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

firecount_file = "data/data/driving_data2425/burnt_area0p05-uk.nc"
BA_file = "../fireMIPbenchmarking/data/benchmarkData/ISIMIP3a_obs/GFED5_Burned_Percentage.nc"
def load_cube(file):
    cube = iris.load_cube(file)
    cube = sub_year_range(cube, [2001, 2021])
    cube = constrain_natural_earth(cube, 'United Kingdom')
    cube = constrain_natural_earth(cube, 'England', shape_cat = 'admin_0_map_units')
    return cube

cube = load_cube(BA_file)
cube_fc = load_cube(firecount_file)
cube_fc.data = cube_fc.data > 0.0

# Compute a 12-month running mean
running_mean_cube = cube.rolling_window('time', iris.analysis.MEAN, 12)

# Compute global mean burnt area over time
global_mean_burnt_area = running_mean_cube.collapsed(['latitude', 'longitude'], 
                                                     iris.analysis.MEAN).data
#set_trace()
plt.style.use("seaborn-v0_8-darkgrid")
global_mean_burnt_area_no_running = cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN).data

# Extract time values and convert to datetime
time_coord = running_mean_cube.coord('time')
time_units = time_coord.units
time_values = [
    datetime.datetime(t.year, t.month, t.day) if isinstance(t, cftime.DatetimeGregorian) else t 
    for t in time_units.num2date(time_coord.points)
]
time_numeric = np.arange(len(global_mean_burnt_area))  # Numeric time axis for regression

# Bayesian Linear Regression (on original scale, not log)
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

# Define the number of knots for the spline
num_knots = 2  
knot_positions = np.linspace(time_numeric.min(), time_numeric.max(), num_knots)

# Create spline basis using patsy
B = patsy.dmatrix(f"bs(time_numeric, knots={knot_positions[1:-1].tolist()}, degree=3, include_intercept=True)", 
                  {"time_numeric": time_numeric}, return_type="dataframe").values
'''
with pm.Model() as model:
    # Priors for spline coefficients
    beta_spline = pm.Normal("beta_spline", mu=0, sigma=10, shape=B.shape[1])

    # Additive model: Sum of spline basis functions weighted by beta coefficients
    mu = pm.Deterministic("mu", pm.math.dot(B, beta_spline))

    # Noise term
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Observed log burnt area
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=np.log(global_mean_burnt_area + 0.001))

    # Sample posterior
    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

# Compute Bayesian trend line and 90% credible interval
trend_median = np.exp(np.median(trace.posterior["mu"].values, axis=(0, 1)))
trend_5 = np.exp(np.percentile(trace.posterior["mu"].values, 5, axis=(0, 1)))
trend_95 = np.exp(np.percentile(trace.posterior["mu"].values, 95, axis=(0, 1)))
'''
# Plot the time series as a bar plot with Bayesian trend
plt.figure(figsize=(10, 5))
plt.bar(time_values, global_mean_burnt_area_no_running[11:], width=20, label="Monthly Burned Area", color="blue", alpha=0.6)
plt.plot(time_values, trend_median, label="Trend (%/year)", color="red", linewidth=2)
plt.fill_between(time_values, trend_5, trend_95, color="red", alpha=0.3, label="90% Credible Interval")

# Formatting
plt.xlabel('Time')
plt.ylabel('Burned Area (%)')
plt.title('UK wide Burned Area')
plt.legend()
plt.savefig("burnt_area_UK_TS.pdf", format="pdf", bbox_inches="tight")
plt.show()
'''
# Compute fire counts per time step
fire_counts = cube_fc.collapsed(['latitude', 'longitude'], iris.analysis.SUM).data

# Plot burnt area time series (bar plot) with Bayesian trend (line)
fig, ax1 = plt.subplots(figsize=(12, 6))

# Left y-axis (Burnt Area)
ax1.bar(time_values, global_mean_burnt_area, width=20, label="Burnt Area 12-month mean", 
        color="royalblue", alpha=0.7, edgecolor="black")
ax1.plot(time_values, trend_median, label="Bayesian Trend", color="darkred", linewidth=2)
ax1.fill_between(time_values, trend_5, trend_95, color="red", alpha=0.2, label="90% Credible Interval")

ax1.set_xlabel('Time', fontsize=14)
ax1.set_ylabel('Burnt Area', fontsize=14, color="royalblue")
ax1.tick_params(axis='y', labelcolor="royalblue")

# Secondary y-axis (Fire Counts)
ax2 = ax1.twinx()
ax2.plot(time_values, fire_counts, label="Fire Count", color="darkgreen", linestyle="dashed", linewidth=2)
ax2.set_ylabel("Fire Count", fontsize=14, color="darkgreen")
ax2.tick_params(axis='y', labelcolor="darkgreen")

# Grid lines & styling
ax1.grid(True, linestyle="--", alpha=0.6)
plt.xticks(rotation=45)
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)

# Legends
ax1.legend(loc="upper left", frameon=False)
ax2.legend(loc="upper right", frameon=False)

plt.title('12-month Running Mean of Burnt Area & Fire Count', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


fire_counts_raw = cube_fc.collapsed(['latitude', 'longitude'], iris.analysis.SUM).data

# Extract time values for fire counts (from cube_fc)
fc_time_coord = cube_fc.coord('time')
fc_time_units = fc_time_coord.units
fc_time_values = [
    datetime.datetime(t.year, t.month, t.day) if isinstance(t, cftime.DatetimeGregorian) else t
    for t in fc_time_units.num2date(fc_time_coord.points)
]

### ---- Plot Burnt Area & Fire Counts ---- ###
fig, ax1 = plt.subplots(figsize=(12, 6))

# Left y-axis (Burnt Area)

ax1.bar(time_values, global_mean_burnt_area_no_running[11:], width=20, label="Monthly Burned Area", color="blue", alpha=0.6)
ax1.plot(time_values, trend_median, label="Bayesian Trend", color="darkred", linewidth=2)
ax1.fill_between(time_values, trend_5, trend_95, color="red", alpha=0.2, label="90% Credible Interval")

ax1.set_xlabel('Time', fontsize=14)
ax1.set_ylabel('Burnt Area', fontsize=14, color="royalblue")
ax1.tick_params(axis='y', labelcolor="royalblue")

# Secondary y-axis (Fire Counts)
ax2 = ax1.twinx()
ax2.plot(fc_time_values, fire_counts_raw, label="Fire Count", color="darkgreen", linestyle="dashed", linewidth=2)
ax2.set_ylabel("Fire Count", fontsize=14, color="darkgreen")
ax2.tick_params(axis='y', labelcolor="darkgreen")

# Grid lines & styling
ax1.grid(True, linestyle="--", alpha=0.6)
plt.xticks(rotation=45)
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)

# Legends
ax1.legend(loc="upper left", frameon=False)
ax2.legend(loc="upper right", frameon=False)

plt.title('12-month Running Mean of Burnt Area & Fire Count', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
'''
