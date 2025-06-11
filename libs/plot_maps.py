import iris
import numpy as np
import cartopy.crs as ccrs

import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pdb import set_trace as browser
from numpy import inf
import numpy as np
import matplotlib.colors as mcolors
import math

import sys
sys.path.append('../../libs/')
sys.path.append('libs/')
sys.path.append('SoW_info/')
import git_info
from to_precision import *
from state_of_wildfires_colours  import SoW_cmap
from pdb import set_trace

def plot_BayesModel_maps(Sim, levels, cmap, ylab = '', Obs = None, 
                         Nrows = 1, Ncols = 2, plot0 = 0, collapse_dim = 'realization',
                         scale = 1, figure_filename = None, set_traceT = False,
                         *args, **kw):
    try:
        if collapse_dim != 'time': Obs = Obs.collapsed('time', iris.analysis.MEAN) 
        
    except:
        pass
    try:
        if collapse_dim != 'time': Obs = Obs.collapsed('time', iris.analysis.MEAN) 
        
    except:
        pass
    try:
        Sim = Sim.collapsed(collapse_dim, iris.analysis.PERCENTILE, percent=[5, 95])
    except:
        set_trace()
    Sim.data[Sim.data <0] = 0.0
    #levels = hist_limits(cube*scale, nlims = 7, symmetrical = False)[0]
    #levels = np.sort(np.append([0], levels[levels > 0.00001]))
    if levels is None:
        levels = np.append(hist_limits(Obs*scale, nlims = 6, symmetrical = False)[0], 
                           hist_limits(Sim*scale, nlims = 6, symmetrical = False)[0])
        levels = np.sort(np.append([0], levels[levels > 0.00001]))
    #set_trace()   
    def plot_map(cube, plot_name, plot_n, **kw2):
        levels = hist_limits(cube*scale, nlims = 6, symmetrical = False)[0]
        plot_annual_mean(cube, levels, cmap, plot_name = plot_name, scale = scale, 
                     Nrows = Nrows, Ncols = Ncols, plot_n = plot_n + plot0, *args, **kw, **kw2)
        
        if plot_n == 1:
            plt.gca().text(-0.1, 0.5, ylab, fontsize=12, rotation=90, va='center', ha='right',
                           transform=plt.gca().transAxes)
    def set_fig_fname(txt):
        if  figure_filename is None:
            return None
        else:
            return figure_filename + '-' + txt + '.nc'
    
    if Obs is None: 
        plot_n = 1
    else:
        plot_map(Obs, "Observations", 1, figure_filename = set_fig_fname('obs'))
        plot_n = 2
    
    plot_map(Sim[0,:], "Simulation -  5%", plot_n, figure_filename = set_fig_fname('-sim05pc'))
    plot_map(Sim[1,:], "Simulation - 95%", plot_n+1, figure_filename = set_fig_fname('-sim95pc'))
    #plot_map(Sim[2,:], "Simulation - 95%", plot_n+2, figure_filename = set_fig_fname('-sim95pc'))
    
    return levels
    
   
def plot_annual_mean(cube, levels, cmap, plot_name = None, scale = None, 
                     Nrows = 1, Ncols = 1, plot_n = 1, colourbar = True, *args, **kw):
    try:
        aa = cube.collapsed('time', iris.analysis.MEAN)
    except:
        aa = cube
    if plot_name is not None: aa.long_name = plot_name
    if scale is not None: aa.data = aa.data * scale
    
    plot_lonely_cube(aa, Nrows, Ncols, plot_n, levels = levels, cmap = cmap, 
                     colourbar = colourbar, grayMask = True, *args, **kw)



def plot_lonely_cube(cube, N = None, M = None, n = None, levels = [0], extend = 'neither', colourbar = True, *args, **kw):

    cf, levels, extend = plot_cube(cube, N,  M, n, levels = levels, extend = extend, *args, **kw)
    if colourbar: 
        addColorbar(cf, levels, extend = extend)
    plt.tight_layout()
    return cf
    
def addColorbar(cf, ticks, *args, **kw):
    cb = plt.colorbar(cf, orientation='horizontal', ticks = ticks, *args, **kw)
    #formatted_ticks = [f'{tick:.2f}' for tick in ticks]
    #cb.ax.set_xticklabels(formatted_ticks)
    return cb

def plot_cube(cube, N, M, n, cmap, levels = None, extend = 'neither', 
             projection = ccrs.Robinson(), grayMask = False, fig = None, 
             figure_filename = None):
    if levels is None:
        levels, extend = hist_limits(cube, levels, 6)

    if n is None:
        ax = plt.axes(projection = projection)
    else:
        if fig is None:
            ax = plt.subplot(N, M, n, projection = projection)
        else:
            ax = fig.add_subplot(N, M, n, projection = projection)
    
    ax.set_title(cube.long_name)

    cmap = plt.get_cmap(cmap)
    
    
    levelsi = [i for i in levels]
    
    if extend == "max" or extend == "both": levelsi += [9E9]
    if extend == "min" or extend == "both": levelsi = [-9E9] + levelsi

    if extend == "max" or extend == "min":
        norm = BoundaryNorm(levelsi, ncolors=cmap.N)
    else:
        norm = BoundaryNorm(levelsi, ncolors=cmap.N)
    
    if grayMask: plt.gca().patch.set_color('.9')
    try:
        cf = iplt.pcolormesh(cube, cmap = cmap, norm = norm) 
    except:
        cf = iplt.pcolormesh(cube, cmap = cmap) 
    
    plt.gca().coastlines()
    if figure_filename is not None:
        try:
            iris.save(cube, figure_filename)
        except:
            set_trace()
    return cf, levels, extend


def plot_cubes_map(cubes, nms, cmap, levels, extend = 'neither',
                   figName = None, units = '', nx = None, ny = None, 
                   cbar_yoff = 0.0, figXscale = 1.0, figYscale = 1.0, 
                   totalMap = None, *args, **kw):
    
    try:
        cubeT =cubes.collapsed('time', totalMap)
        nms = [i for i in nms]
        nms.append('Total')
    except: cubeT = None  

    try: cubes = [cubes[i] for i in range(0, cubes.shape[0])]
    except: pass
    
    if cubeT is not None: cubes.append(cubeT)
    
    for i in range(0, len(cubes)):  cubes[i].long_name = nms[i]
    nplts = len(cubes)
    if nx is None and ny is None:
        nx = int(math.sqrt(nplts))
        ny = math.ceil(nplts / float(nx))
        nx = nx + 1.0
    elif nx is None:   
        nx = math.ceil(nplts / float(ny)) + 1
    elif ny is None:
        ny = math.ceil(nplts / float(nx))
    
    plt.figure(figsize = (nx * 2 * figXscale, ny * 4 * figYscale))

    for i in range(0, len(cubes)):         
        cmapi = cmap if (type(cmap) is str) else cmap[i]
        cf = plot_cube(cubes[i], nx, ny, i + 1, cmapi, levels, extend, *args, **kw)

    colorbar_axes = plt.gcf().add_axes([0.15, cbar_yoff + 0.5 / nx, 0.7, 0.15 / nx])
    cb = addColorbar(cf, levels, colorbar_axes, extend = extend)
    cb.set_label(units)

    plt.tight_layout()
    if (figName is not None):
        if figName == 'show':
            plt.show()
        else :
            print(figName)
            git = 'rev:  ' + git_info.rev + '\n' + 'repo: ' + git_info.url
            plt.gcf().text(.05, .95, git, rotation = 270, verticalalignment = "top")
            plt.savefig(figName, bbox_inches='tight')
            plt.clf()

def hist_limits(dat, lims = None, nlims = 5, symmetrical = True):
    def select_lims(prec, nlims):
        nlims0 = nlims
        for p in range(0,100 - nlims0):
            nlims = nlims0 + p
            try:
                lims  = np.percentile(dat.data[~np.isnan(dat.data)].data, range(0, 100, int(100/nlims)))
            except:
                set_trace()
            
            if (lims[0]==-inf): lims.pop(0)
            
            lims = [to_precision(i, prec) for i in lims]
            lims = np.unique(lims)
            if (len(lims) >= nlims0): break
        return lims

    if (lims is None):
        for prec in range(1,5):
            lims = select_lims(prec, nlims)
            print(lims)
            if len(lims) > nlims: break
        
        new_lims = True
    else:
        new_lims = False
    if (lims[0] < 0.0):
        if (new_lims): 
            # are the more levels less than zero or gt  then zero  
            if (sum(i < 0.0 for i in lims) > sum(i > 0.0 for i  in lims)):
                # if more gt zero
                lims = [i for i in lims if i < 0.0]
                lims = np.concatenate((lims,[-i for i in lims[::-1]]))  

            else:
                # if more lt zero
                lims = [i for i in lims if i > 0.0]
                lims = np.concatenate(([-i for i in lims[::-1]], lims))     
        extend = 'both'
        
    else:
        extend = 'max'

    if len(lims) == 1: 
        lims = [-0.0001, -0.000001, 0.000001, 0.0001] if lims == 0.0 else [lims[0] * (1 + i) for i in [-0.1, -0.01, 0.01, 0.1]]
        extend = 'neither'

    if np.log10(lims[0]/lims[1]) > 3: lims[0] = 1000 * lims[1]
    if np.log10(lims[-1] / lims[-2]) > 3: lims[-1] = 1000 * lims[-2]
    if len(lims) < 4:
        lims = np.interp(np.arange(0, len(lims), len(lims)/6.0), range(0, len(lims)), lims)
    
    return (lims, extend)

import numpy as np

def concat_cube_data(cubes):
    """
    Concatenate data from a list of Iris cubes into one flat NumPy array,
    skipping any masked/invalid data.
    """
    data_list = []

    for cube in cubes:
        data = np.ma.masked_invalid(cube.data)
        if np.ma.is_masked(data):
            valid_data = data.compressed()  # only unmasked values
        else:
            valid_data = data.ravel()
        data_list.append(valid_data)

    return np.concatenate(data_list)


def auto_pretty_levels(data, n_levels=7, log_ok=True, ratio = None, force0 = False):
    """
    Generate 'pretty' contour levels that break the data into roughly equal-sized areas.

    Parameters:
    - data: array-like or Iris cube (flattened automatically)
    - n_levels: desired number of contour bins (not including endpoints)
    - log_ok: allow log scaling if data is skewed and positive

    Returns:
    - levels: list of nicely rounded level values
    """
    print(ratio)
    try:
        data = concat_cube_data(data)
    except:
        try:
            data = data.data
        except:
            pass
    # Flatten data and mask NaNs
    try:
        data = np.ma.masked_invalid(np.ravel(data))
    except:
        set_trace()
    if data.mask.all():
        raise ValueError("No valid data found to calculate levels.")

    # Try log-scale if distribution is strongly right-skewed and positive
    try:
        yay = log_ok and np.all(data > 0) and (np.percentile(data, 90) / np.percentile(data, 10)) > 50
    except:
        set_trace()
    if log_ok and np.all(data > 0) and (np.percentile(data, 90) / np.percentile(data, 10)) > 50:
        log_data = np.log10(data)
        percentiles = np.linspace(0, 100, n_levels + 1)
        levels_raw = np.power(10, np.percentile(log_data, percentiles))
    else:
        percentiles = np.linspace(0, 100, n_levels + 1)
        levels_raw = np.percentile(data, percentiles)

    # Round levels to 'nice' numbers
    def nice_round(x):
        if x == 0:
            return 0
        magnitude = 10 ** np.floor(np.log10(abs(x)))
        mantissa = round(x / magnitude, 1)
        return mantissa * magnitude

    levels_rounded = sorted(set([nice_round(lv) for lv in levels_raw]))

    # Ensure levels are strictly increasing and unique
    while len(levels_rounded) <= 2 and n_levels < 20:
        n_levels += 2  # try with more bins if too few unique rounded levels
        return auto_pretty_levels(data, n_levels=n_levels, log_ok=log_ok)
    
    levels_rounded = np.array(levels_rounded)
    levels_rounded0 = levels_rounded.copy()
    if ratio is not None:
        levels_rounded[levels_rounded > 1000] = 1000
        levels_rounded[levels_rounded < 1/1000.0] = 1/1000.0
        levels_rounded = np.log(levels_rounded) / ratio
        data =  np.log(data ) / ratio
        
    if force0 or (any(levels_rounded < 0) and any(data > 0)) or \
            (any(levels_rounded > 0) and any(data < 0)):
        levels_rounded = np.sort(np.unique(np.append(levels_rounded, - levels_rounded)))
        
    if ratio is not None:
        levels_rounded = np.exp(levels_rounded*100)
        
        try:
            levels_rounded = np.vectorize(nice_round)(levels_rounded)
        except:
            set_trace()
        levels_rounded = np.unique(levels_rounded)
    print(levels_rounded)
    return levels_rounded

def add_overlay_value(cube, value, col, ax):
    '''
    import matplotlib.colors as mcolors
    
    ax.imshow((cube.data == 0), origin='lower', cmap=mcolors.ListedColormap(['none', 'black']), alpha=0.3,
          extent=[cube.coord('longitude').points.min(), cube.coord('longitude').points.max(),
                  cube.coord('latitude').points.min(), cube.coord('latitude').points.max()])
    '''
    


    # --- Create a binary mask for where cube == 0 ---
    zero_mask = (cube.data == value)
    
    # Create a new cube with 1 where data == 0, masked elsewhere
    import copy
    highlight_cube = copy.deepcopy(cube)
    highlight_cube.data = zero_mask.astype(float)
    highlight_cube.data[~zero_mask] = np.nan  # Mask non-zero

    # --- Define custom colormap: fully transparent except where == 1 ---
    highlight_cmap = mcolors.ListedColormap(['none', col])  # 'red' can be any color
    
    # Make norm so that only 1 maps to red
    highlight_norm = mcolors.BoundaryNorm([0, 0.5, 1.5], ncolors=2)
    
    # --- Overlay the mask ---
    iplt.contourf(highlight_cube, levels=[0.5, 1.5], cmap=highlight_cmap, norm=highlight_norm, axes=ax, add_colorbar=False)


def get_cube_extent(cube):
    lon_min = cube.coord('longitude').points.min()
    lon_max = cube.coord('longitude').points.max()
    lat_min = cube.coord('latitude').points.min()
    lat_max = cube.coord('latitude').points.max()
    return [lon_min, lon_max, lat_min, lat_max]

def set_up_sow_plot_windows(n_rows, n_cols, eg_cube, figsize = None):
    if figsize is None:
        figsize = (n_rows * 16/5, n_cols*4)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16), 
                             subplot_kw={'projection': ccrs.PlateCarree()})

    extent = get_cube_extent(eg_cube)
    extent[0] -= (extent[1] - extent[0])*0.1
    extent[1] += (extent[1] - extent[0])*0.1
    extent[2] -= (extent[3] - extent[2])*0.1
    extent[3] += (extent[3] - extent[2])*0.1
    
    for ax in axes.flat:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Flatten axes for easy indexing
    axes = axes.flatten()
    return axes
    

def plot_map_sow(cube, title='', contour_obs=None, cmap='RdBu_r', 
             levels = None, extend = 'both', ax=None,
             cbar_label = '', overlay_value = None, overlay_col = "#cfe9ff"):
    
    cube.long_name = title
    cube.rename(title)
    is_catigorical =  np.issubdtype(cube.core_data().dtype, np.integer)
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))
    # Main filled contour
    if levels is  None:
        levels = auto_pretty_levels(cube.data)
    if is_catigorical:
        norm = BoundaryNorm(boundaries=np.array(levels) + 0.5, ncolors=cmap.N)
    else:   
        norm = BoundaryNorm(boundaries=levels,  ncolors=cmap.N, extend = extend)
    
    
    img = iplt.contourf(cube, levels=levels, cmap=cmap, axes=ax, extend = extend, 
                        norm = norm)
    if overlay_value is not None:
        add_overlay_value(cube, overlay_value, overlay_col, ax)

    if is_catigorical:
        tick_positions = np.array(levels) + 0.5
        tick_labels = [str(level) for level in levels]
        cbar = plt.colorbar(img, ax=ax, orientation='horizontal',
                            ticks=tick_positions)
        cbar.ax.set_xticklabels(tick_labels) 
    else:
        cbar = plt.colorbar(img, ax=ax, ticks=levels, orientation='horizontal')
    cbar.set_label(cbar_label, labelpad=10, loc='center')
    cbar.ax.xaxis.set_label_position('top')
     
    # Add boundaries
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    # Optional observed burned area anomaly contour
    if contour_obs is not None:
        qplt.contour(contour_obs, levels=[0], colors='#8a3b00', linewidths=1, axes=ax)

    print(title)
    ax.set_title(title)
    return img

