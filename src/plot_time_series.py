import iris
import iris.coord_categorisation
import iris.plot as iplt
from iris.analysis.cartography import area_weights
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdb import set_trace

# Load the cube
cube = iris.load_cube("data/data/driving_data2425/LA/burnt_area.nc")
try:
    cube.coord('longitude').guess_bounds()
except:
    pass
try:
    cube.coord('latitude').guess_bounds()
except:
    pass
# Calculate grid cell area weights


weights = area_weights(cube)

# Collapse to time series (weighted average over lat/lon)
mean_cube = cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN, weights=weights)

# Convert time points to decimal years
def to_decimal_year(time_coord):
    import cftime
    times = time_coord.units.num2date(time_coord.points)
    decimal_years = [t.year + (t.timetuple().tm_yday - 1) / (366 if t.year % 4 == 0 else 365) for t in times]
    return np.array(decimal_years)

time_coord = mean_cube.coord('time')
decimal_years = to_decimal_year(time_coord)
area_weighted_series = mean_cube.data


# Read CSV while skipping comments
df = pd.read_csv(
    "outputs/outputs/ConFire_LA-2425-attempt1/figs/_15-frac_points_0.2-baseline-control_TS/time_series-Evaluate-0.0.csv",
    comment='#',
    names=["year", "p5", "p10", "p25", "p75", "p90", "p95"]
)
percentile_cols = ["p5", "p10", "p25", "p75", "p90", "p95"]
df[percentile_cols] = df[percentile_cols] * 100
years_csv = df["year"].values

plt.figure(figsize=(10, 6))

# Plot percentiles as shaded bands
plt.fill_between(df["year"], df["p25"], df["p75"], color='gray', alpha=0.3, label='25–75%')
plt.fill_between(df["year"], df["p10"], df["p90"], color='gray', alpha=0.2, label='10–90%')
plt.fill_between(df["year"], df["p5"], df["p95"], color='gray', alpha=0.1, label='5–95%')

# Plot your time series
plt.plot(decimal_years, area_weighted_series, color='red', label='Area-weighted burnt area')

plt.xlabel("Year")
plt.ylabel("Burnt area (units?)")
plt.title("Burnt Area Time Series vs Control Percentiles")
plt.legend()
plt.tight_layout()
plt.show()

set_trace()
