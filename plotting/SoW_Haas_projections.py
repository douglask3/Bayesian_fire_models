import sys
sys.path.append('.')
sys.path.append('plotting/')
sys.path.append('src/')
sys.path.append('src/attribution/')
sys.path.append('SoW_info/')
from  pathlib import Path
from  attribution_where import *
from plot_maps import *
from constrain_cubes_standard import *
import numpy as np
from pdb import set_trace
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import numpy as np
import iris
import iris.coords
import iris.analysis

def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def upscale_cube_bilinear(cube, factor=5):
    """
    Upscale an Iris cube by splitting each grid cell into (factor x factor)
    using bilinear interpolation.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube with regular lat/lon grid.
    factor : int
        Upscaling factor (e.g., 5 → each cell becomes 5x5)

    Returns
    -------
    iris.cube.Cube
        Regridded cube at higher resolution
    """

    # Extract coordinates
    lat = cube.coord('latitude')
    lon = cube.coord('longitude')

    # Assume regular grid spacing
    dlat = np.mean(np.abs(np.diff(lat.points)))
    dlon = np.mean(np.diff(lon.points))

    # New higher-resolution spacing
    new_dlat = dlat / factor
    new_dlon = dlon / factor

    # Build new coordinate arrays (same extent)
    new_lat = np.arange(lat.points.min(),
                        lat.points.max() + new_dlat,
                        new_dlat)

    new_lon = np.arange(lon.points.min(),
                        lon.points.max() + new_dlon,
                        new_dlon)
    #set_trace()
    # Create new coords
    new_lat_coord = iris.coords.DimCoord(
        new_lat,
        standard_name='latitude',
        units=lat.units
    )

    new_lon_coord = iris.coords.DimCoord(
        new_lon,
        standard_name='longitude',
        units=lon.units
    )

    # Create target cube (empty data, just grid definition)
    target_cube = iris.cube.Cube(
        np.zeros((len(new_lat), len(new_lon))),
        dim_coords_and_dims=[(new_lat_coord, 0),
                             (new_lon_coord, 1)]
    )

    # Regrid using bilinear interpolation
    regridded = cube.regrid(target_cube, iris.analysis.Linear())

    return regridded

def plot_agreement_map(region_agree, ax, cmap = SoW_cmap["diverging_TealGreyOrange"],
                       hist_loc = "upper left"):
    plot_map_sow(region_agree, add_cbar = False, extend = 'neither', 
                 levels = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
                 cmap = cmap, use_pcolmesh = True, ax = ax,
                    is_catigorical = True,
                 cbar_orientation = 'horizontal')
    
    counts = region_agree.data.flatten().astype('int')
    counts = np.bincount(counts[counts>=0], minlength=5)
    
    ax_inset = inset_axes(
        ax,
        width="15%",   # width relative to main axis
        height="15%",  # height relative to main axis
        loc=hist_loc,
        borderpad=0.01
    )
    ax_inset.patch.set_alpha(0)
    norm = mcolors.Normalize(vmin=0, vmax=4)
    
    ax_inset.bar(range(5), counts, color = [cmap(norm(i)) for i in range(5)])

    ax_inset.set_xticks(range(5))
    ax_inset.set_yticks([])
    ax_inset.spines['left'].set_visible(False)
    ax_inset.spines['right'].set_visible(False)
    ax_inset.spines['top'].set_visible(False)

    
    ax_inset.tick_params(axis='both', labelsize=8)
    ax_inset.set_ylim(0, counts.max() * 1.1)


    
def for_variable_rcp(variable, longname, rcp, region, ax,
                     shapefile_path, dir, baseline, future, experiment, 
                     return_eg = False, plot_baseline = False, 
                     cbar_orientation = 'horizontal', *args, **kw):
    base_file = dir + baseline + variable + '.nc'

    def load_regid(file):
        base = iris.load_cube(file)
        
        base = contrain_to_sow_shapefile(base, shapefile_path, 
                                         region)
        return base#upscale_cube_bilinear(base)

    base = load_regid(base_file)
    
    if plot_baseline:
        if variable == 'BA':
            base *= 100
            levels = signif(auto_pretty_levels(base.data), 1)
        else:
            levels = None
        plot_map_sow(base, add_cbar = True, extend = 'max',
                 cmap = SoW_cmap['gradient_red'], use_pcolmesh = True, ax = ax,
                 cbar_orientation = cbar_orientation, levels = levels)
        #ax.set_title('baseline, ' + variable)
        ax.annotate(longname, xy=(-0.1, 0.5),  rotation=90, 
                    fontsize=14,  # <-- Change text size here (in points)
                    weight="bold",  # Optional: makes it bold like a standard title
                    xycoords="axes fraction", ha="center", va="center")
        if variable == 'BA':
            ax.annotate("Baseline", xy=(0.5, 1.1), 
                    fontsize=14,  # <-- Change text size here (in points)
                    weight="bold",  # Optional: makes it bold like a standard title
                    xycoords="axes fraction", ha="center", va="center")
        return None
        
    
    futr_dir = dir + '/' + future + '/' + variable + experiment  +'/'
    files = glob.glob(futr_dir + '*' + rcp + '*.nc')
    
    agree = base.copy()
    agree.data[:] = 0.0
    tot = base.copy()
    for file in files:
        futr = load_regid(file)
        agree.data[futr.data >= base.data] += 1
        tot.data += futr.data
    
    agree.data[base.data.mask] = np.nan
    agree.data[tot.data == 0] = np.nan
    
    if return_eg:
        return agree
    plot_agreement_map(agree, ax, *args, **kw)
    #ax.set_title(rcp + ', ' + variable)
    if variable == 'BA':
        ax.annotate(rcp, xy=(0.5, 1.1),  
                    fontsize=14,  # <-- Change text size here (in points)
                    weight="bold",  # Optional: makes it bold like a standard title
                    xycoords="axes fraction", ha="center", va="center")


if __name__=="__main__":
    dir = "data/data/driving_data2526/Haas/"
    baseline = "baseline_"
    future = "future_projections"
    experiment = "_climate_co2"
    variables = ["BA", "FS", "FI"]
    longnames = ["Burned Area", "Fire Size", "Fire Intensity"]
    rcps = ["RCP26", "RCP60"]
    shapefile_path ="data/data/driving_data2526/Focal_regions/SoW2526_Focal_MASTER_20260218.shp"
    regions = ["Northwest Iberia", "Midwestern Canadian Shield forests", 
                "Chilean Temperate Forests and Matorral"]

    hist_locs = ["lower right", "lower left", "upper left"]
    size_scales = [4.5, 4.5, 1.5]
    cbar_orientations = ['horizontal', 'horizontal', 'vertical']
    #region = regions[0]
    for region, hist_loc, size_scale, cbar_orientation in \
            zip(regions, hist_locs, size_scales, cbar_orientations):
        eg_cube = for_variable_rcp(variables[0], longnames[0], rcps[0], region, None,
                                 shapefile_path, dir, baseline, future, experiment, True, 
                                 cbar_orientation=cbar_orientation)
        
        fig, axes = set_up_sow_plot_windows(3, 3, eg_cube,  size_scale = size_scale, 
                                            oma = [0.5, 0.5, 0.5, 0.1])
        i = 0
        for variable, longname in zip(variables, longnames):
                    
            for_variable_rcp(variable, longname, rcps[0], region, axes[i],
                             shapefile_path, dir, baseline, future, experiment,
                             plot_baseline = True, cbar_orientation=cbar_orientation)
            i += 1
            for rcp in rcps:    
                for_variable_rcp(variable, longname, rcp, region, axes[i],
                                 shapefile_path, dir, baseline, future, experiment,
                                 hist_loc = hist_loc, cbar_orientation=cbar_orientation)
                i += 1
        
        plt.savefig('figs/Haas-' + region + '.png')
    set_trace()
    
