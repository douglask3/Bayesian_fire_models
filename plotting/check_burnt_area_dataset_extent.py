import iris
from iris.coord_categorisation import add_month
import iris.analysis.cartography
import iris.plot as iplt
import matplotlib.pyplot as plt
import datetime as dt
from pdb import set_trace
import sys
sys.path.append('libs/')
from constrain_cubes_standard import *
from plot_maps import *
import seaborn as sns
from iris.coord_systems import GeogCS
import calendar
from state_of_wildfires_region_info  import get_region_info
years = [2003, 2019]

gfed_path = "/data/users/douglas.kelley/fireMIPbenchmarking/data/benchmarkData/ISIMIP3a_obs/GFED5_Burned_Percentage.nc"

def run_for_region(region):
    region_info = get_region_info(region)[region]
    
    rname = region_info['longname']
    rdir = region_info['dir']
    mnths = region_info['mnths']
    # Load cubes
    cube1 = iris.load_cube(gfed_path)
    cube2 = iris.load_cube("data/data/driving_data2425/" + rdir + "/burnt_area.nc")

    cube1 = constrain_to_time(cube1, years, mnths)
    cube2 = constrain_to_time(cube2, years, mnths)
    
    lat2 = cube2.coord('latitude')
    lon2 = cube2.coord('longitude')
    
    lat_constraint = iris.Constraint(latitude=lambda x: lat2.points.min() <= x <= lat2.points.max())
    lon_constraint = iris.Constraint(longitude=lambda x: lon2.points.min() <= x <= lon2.points.max())

    cube1 = cube1.extract(lat_constraint & lon_constraint)

    wgs84 = GeogCS(6371229)  # Iris's default Earth radius for WGS84-like sphere

    for cube in [cube1, cube2]:
        for coord in [cube.coord('latitude'), cube.coord('longitude')]:
            coord.coord_system = wgs84

    cube2 = cube2.regrid(cube1, iris.analysis.Linear())
    a1 = cube1.collapsed(['time'], iris.analysis.MEAN) * len(mnths)
    a2 = cube2.collapsed(['time'], iris.analysis.MEAN) * len(mnths)

    fig, axes = set_up_sow_plot_windows(2, 2, a2, size_scale = 7)

    a1.data.mask  = a2.data.mask
    plot_map_sow(a2, "MCD64A1", 
                 cmap=SoW_cmap['gradient_red'], 
                 levels=[0, 0.1, 0.2, 0.5, 1, 2, 5, 10],extend = 'max', 
                 ax=axes[0], cbar_label = "Burned Area (%)",
                 overlay_value = 0.0)

    plot_map_sow(a1, "GFED5", 
                 cmap=SoW_cmap['gradient_red'], 
                 levels=[0, 0.1, 0.2, 0.5, 1, 2, 5, 10],extend = 'max', 
                 ax=axes[1], cbar_label = "Burned Area  (%)",
                 overlay_value = 0.0)

    diff = a2.copy()
    diff.data -= a1.data
    plot_map_sow(diff, "MCD64A1 - GFED5", 
                 cmap=SoW_cmap['diverging_TealOrange'], 
                 ax=axes[2], cbar_label = "Differnce in Burned Area (%)",
                 overlay_value = 0.0, overlay_col = "#ffffff")
    
    # --- Flatten and mask ---
    data1 = cube1.data.flatten()
    data2 = cube2.data.flatten()
    
    valid = np.isfinite(data1) & np.isfinite(data2)
    x = data2[valid]
    y = data1[valid]
    
    # --- Main scatter plot ---
    ax = axes[3]
    sns.scatterplot(x=x, y=y, s=10, alpha=0.5, edgecolor=None, ax=ax)

    # --- 1:1 reference line ---
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'k--', lw=1, label='1:1 line')
    
    # --- Best fit line (OLS regression) ---
    m, c = np.polyfit(x, y, 1)
    xp = np.array([0.0000000001, 100])
    
    ax.plot(xp, m * xp + c, color='red', lw=1.5, label=f'Best fit') #: y={m:.2f}x+{c:.2f}
    
    # --- Labels and formatting ---
    ax.set_xlabel("MCD64A1: Burned %")
    ax.set_ylabel("GFED5: Burned %")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    #ax.set_aspect('equal')
    ax.legend()

    # custom ticks
    ticks =  np.arange(0, max(x.max(), y.max()) +5, 10)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    rname += ' - ' + str(years[0]) + ' to ' + str(years[1]) + ', ' 
    rname += calendar.month_name[int(mnths[0])-1] 
    if len(mnths) > 1:
        for mn in mnths[1:]:
            rname += ', ' + calendar.month_name[int(mnths[mn])-1] 
    fig.suptitle(rname, fontsize=16)
    plt.tight_layout()
    plt.savefig("figs/burnt_area_product_assessment-" + rdir + ".png",  dpi=300)

run_for_region("Congo")
set_trace()
