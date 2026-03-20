from pdb import set_trace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import gaussian_kde
from scipy.integrate import simpson as simps

# --- Load CSVs ---
factual_csv = "outputs/outputs/ConFire_LA-2425-attempt2/figs/_15-frac_points_0.2-factual-control_TS/mean/points-Evaluate.csv"
counterfactual_csv = "outputs/outputs/ConFire_LA-2425-attempt2/figs/_15-frac_points_0.2-counterfactual-control_TS/mean/points-Evaluate.csv"

factual_df = pd.read_csv(factual_csv, header=None)
counterfactual_df = pd.read_csv(counterfactual_csv, header=None)

# --- Winter months only ---
winter_indices = factual_df.index[factual_df.index % 12 <= 1 ]
factual_winter = factual_df.loc[winter_indices].values.flatten()*100
counterfactual_winter = counterfactual_df.loc[winter_indices].values.flatten()*100

# --- Load burnt area data ---
ds = xr.open_dataset("data/data/driving_data2425/LA/burnt_area.nc")
burnt_var_name = list(ds.data_vars)[1]
burnt_area = ds[burnt_var_name]
burnt_mean = float(burnt_area.isel(time=288).mean().values)

# --- Pseudo-log transform ---
eps = 1e-3
factual_log = np.log10(factual_winter + eps)
counterfactual_log = np.log10(counterfactual_winter + eps)
burnt_mean_log = np.log10(burnt_mean + eps)

# --- KDE fit ---
kde_f = gaussian_kde(factual_log)
kde_c = gaussian_kde(counterfactual_log)

# --- X range for density curves ---
x_dense = np.linspace(min(factual_log.min(), counterfactual_log.min(), burnt_mean_log) - eps,
                      max(factual_log.max(), counterfactual_log.max(), burnt_mean_log) + eps, 1000)

f_dens = kde_f(x_dense)
c_dens = kde_c(x_dense)

# --- Ratio above threshold using KDE ---
mask = x_dense > burnt_mean_log
set_trace()
area_f = simps(f_dens[mask], x_dense[mask])
area_c = simps(c_dens[mask], x_dense[mask])
from pdb import set_trace
set_trace()
ratio = area_f / area_c if area_c != 0 else np.inf
print(ratio)
# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(x_dense, f_dens, color='red', lw=2, label='Factual (KDE)')
plt.plot(x_dense, c_dens, color='blue', lw=2, label='Counterfactual (KDE)')

plt.axvline(burnt_mean_log, color='black', linestyle='dashed', label=f'Mean burnt area (log10): {burnt_mean_log:.2f}')

# --- Tick labels in original units ---
tick_vals = np.array([0.001, 0.01, 0.1, 1, 10, 100])
tick_locs = np.log10(tick_vals + eps)
plt.xticks(ticks=tick_locs, labels=[f"{v:.3g}" for v in tick_vals])
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Winter Distributions (KDE): Factual vs Counterfactual")

# --- Annotate ratio ---
plt.text(burnt_mean_log + 0.1, 0.95 * max(max(f_dens), max(c_dens)),
         f'KDE ratio above threshold (F/C): {ratio:.2f}', fontsize=12)

# --- Grid, legend, layout ---
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

