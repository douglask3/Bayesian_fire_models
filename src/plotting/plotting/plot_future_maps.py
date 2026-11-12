import sys
sys.path.append('.')
sys.path.append('plotting/')
sys.path.append('src/')
sys.path.append('src/attribution/')
sys.path.append('SoW_info/')
from  pathlib import Path
from  attribution_where import *
from plot_change_in_burned_area import *
from plot_maps import *
from constrain_cubes_standard import *
import numpy as np
import difflib
import hashlib



def plot_change_in_burned_area_future(dir_f, dir_cf, obs_file = None,
                               year = None, cfyear = None, 
                               af_title = "Amplification factor",
                               af_title2 = "",
                               pval_title = "Likelihood",
                               mnths = None, inverse_af = True, max_BA = False, 
                               af_cmap = "gradient_red", axes = None):

    if cfyear is None:
        cfyear = year
    def open_cube(filename, year):
        cube = iris.load_cube(filename)
        cube0 = cube.copy()
        
        if not isinstance(year, list):
            year = [year]
        cube = sub_year_range(cube, year)
        if mnths is not None: cube = sub_year_months(cube, mnths)
        
        if  mnths is None or len(mnths) > 1: 
            if max_BA:
                cube = cube.collapsed('time', iris.analysis.MAX)
            else:
                cube = cube.collapsed('time', iris.analysis.MEAN)        
        return cube
    obs = open_cube(obs_file, year)
    
    ffiles = glob.glob(dir_f + "**/*", recursive=True)
    cffiles = glob.glob(dir_cf + "**/*", recursive=True)

    def fileID(file, dir):
        return ''.join(''.join(file.split(dir.split('*')[0])).split(dir.split('*')[1]))

    fid = [fileID(file, dir_f) for file in ffiles]
    cfid = [fileID(file, dir_cf) for file in cffiles]
    
    nc_files = set(fid) & set(cfid)
    nc_files = [file for file in nc_files if 'sample-pred' in file]
    nc_files.sort()

    diff = difflib.ndiff(dir_f, dir_cf)
    diff = ''.join( char[2] for char in diff if char.startswith('+ '))
    temp_file = 'plot_future_maps.py' + dir_f + diff
    if inverse_af: temp_file += "inverse_af"
    if max_BA: temp_file += "max_BA"
    if year is not None:
        temp_file += 'fyr' + '_'.join([str(yr) for yr in year])
    if cfyear is not None:
        temp_file += 'cfyr' + '_'.join([str(yr) for yr in cfyear])
    if mnths is not None:
        temp_file += 'cfyr' + '_'.join([str(mn) for mn in mnths])
    
    
    def amplifcation_from_file(file_id, obs):
        ffile =  str(np.array(ffiles)[np.array(fid) == file_id][0])
        cffile =  str(np.array(cffiles)[np.array(cfid) == file_id][0])
        temp_out = temp_file + ffile
        temp_out = 'temp2/hash' + hashlib.sha256(temp_out.encode("utf-8")).hexdigest()
        temp_out_f = temp_out + '-f.nc'
        temp_out_cf = temp_out + '-cf.nc'
        temp_out_af = temp_out + '-af.nc'
        temp_out_prob = temp_out + '-prob.nc'
        temp_out_direction = temp_out + '-direction.nc'
        try:
            out = [iris.load_cube(temp_out_f),
                   iris.load_cube(temp_out_cf),
                   iris.load_cube(temp_out_af),
                   iris.load_cube(temp_out_prob),
                   iris.load_cube(temp_out_direction)]
        except:
            f_cube = open_cube(ffile, year)
            try:
                cf_cube = open_cube(cffile, cfyear)
            except:
                set_trace()
            af_cube = f_cube.copy()
            af_cube.data /= cf_cube.data
            if inverse_af:  af_cube.data = 1.0/af_cube.data
        
            prob = f_cube.copy()
            prob.data = np.exp(obs.data*np.log(f_cube.data) + \
                                (1.0-obs.data)*np.log((1-f_cube.data)))

            direction = af_cube.copy()
            direction.data = direction.data > 1 
            direction.data = direction.data.astype('float32')
            
            iris.save(f_cube, temp_out_f)
            iris.save(cf_cube, temp_out_cf)
            iris.save(af_cube, temp_out_af)
            iris.save(prob, temp_out_prob)
            iris.save(direction, temp_out_direction)
            
            out = [f_cube, cf_cube, af_cube, prob, direction]
        return out


    temp_out = 'temp2/hash2' + hashlib.sha256(temp_file.encode("utf-8")).hexdigest()
    temp_out_af05 = temp_out + '-af05.nc'
    temp_out_af50 = temp_out + '-af50.nc'
    temp_out_af95 = temp_out + '-af95.nc'
    temp_out_pval = temp_out + '-pval.nc'
    temp_out_infr = temp_out + '-infr.nc'

    try:
        af05 = iris.load_cube(temp_out_af05)
        af50 = iris.load_cube(temp_out_af50)
        af95 = iris.load_cube(temp_out_af95)
        infr = iris.load_cube(temp_out_infr)
        pval = iris.load_cube(temp_out_pval)
    except:
        cubes = np.array([amplifcation_from_file(file, obs) for file in nc_files])
        
        eg_cube = cubes[0][0].copy()
        cdata = np.array([[cube.data for cube in cubesi] for cubesi in cubes])
        cdata[cdata>9E9] = np.nan    
    
        af05 = eg_cube.copy()        
        af05.data = np.nanpercentile(cdata[:,2,:,:], [5], axis=0)[0]

        af50 = eg_cube.copy()        
        af50.data = np.nanpercentile(cdata[:,2,:,:], [50], axis=0)[0]

        af95 = eg_cube.copy()        
        af95.data = np.nanpercentile(cdata[:,2,:,:], [95], axis=0)[0]
                #, weights=cdata[:,3,:,:], method="inverted_cdf")
        infr = eg_cube.copy()  
        infr.data = np.nanmean(cdata[:,1,:,:], axis = 0)/ np.nanmean(cdata[:,0,:,:], axis = 0)      
        pval = eg_cube.copy()
        pval.data = 100*np.nanmean(cdata[:,4,:,:], axis = 0)

        iris.save(af05, temp_out_af05)
        iris.save(af50, temp_out_af50)
        iris.save(af95, temp_out_af95)
        iris.save(infr, temp_out_infr)
        iris.save(pval, temp_out_pval)
       
    obs_c = obs.copy()
    ba95 = np.sort(obs.data.data[obs.data.data <100])#
    ba95 = ba95[np.where(ba95.cumsum()>(0.05*ba95.sum()))[0][0]]
        
    obs_c.data[:] = obs_c.data> ba95
        
    if axes is None:
        fig, axes = set_up_sow_plot_windows(1, 2, obs,  
                                            size_scale = 2 + obs.shape[1]/obs.shape[0],         
                                            oma = [0.6, 0.1, 0.25, 0.25])

    levels = [0, 1/2, 1/1.75, 1/1.5,  1/1.25, 1,  1.25, 1.5, 1.75, 2]
    tick_labels = ['0', '1/2', '4/7', '2/3',  '4/5', '1', '5/4', 
                   '3/2', '7/4', '2']
    #levels = [0, 1, 1.1, 1.5, 2, 5]
    #tick_labels = ['0', 'no\nchange', 
    #               '1.1', '1.5', '2', '5']
    af05.data[obs.data.mask] = np.nan
    af95.data[obs.data.mask] = np.nan
    pval.data[obs.data.mask] = np.nan
    plot_map_sow(af05, af_title, 
                 scatter_obs = obs_c,
                 add_cbar = True, extend = 'max',
                 levels = levels, tick_labels = tick_labels,
                 cmap=SoW_cmap[af_cmap], use_pcolmesh = True, ax = axes[0],
                 cbar_orientation = 'horizontal', 
                 cbar_lab_rotate = 45, cbar_top_and_bottom = True)

    
    plot_map_sow(af95, af_title2, 
                 scatter_obs = obs_c,
                 add_cbar = True, extend = 'max',
                 levels = levels, tick_labels = tick_labels,
                 cmap=SoW_cmap[af_cmap], use_pcolmesh = True, ax = axes[1],
                 cbar_orientation = 'horizontal', 
                 cbar_lab_rotate = 45, cbar_top_and_bottom = True)
    
    levels = [0, 1, 10, 33, 66, 90, 99, 100]
    range_edges = np.arange(8)*100/7
    top_tick_pos = range_edges[1:].copy() - 100/14

    top_tick_labels = ["Extremely\nUnlikely", "Very\nUnlikely", "Unlikely", 
                       "As likely\nas not","Likely", "Very\nLikely", 
                                                 "Virtually\nCertain"]
        
    img = plot_map_sow(pval, pval_title, scatter_obs = obs_c,
                        cmap=SoW_cmap['confidence_hues'], 
                        levels = levels,
                        extend = 'neither', cbar_label = "",
                        add_cbar = False, use_pcolmesh = True,
                        ax = axes[2], cbar_orientation = 'horizontal', 
                        cbar_lab_rotate = 45, cbar_top_and_bottom = True)

    add_attribubtion_map_cbar(img, axes[2], levels, 
                              top_tick_labels= top_tick_labels, 
                              range_edges = range_edges, top_tick_pos = top_tick_pos,
                              cbar_label = '')
    #set_trace()

    
         

dir_f = "outputs/outputs_scratch/UK/isimip/test-ukSPECIFIC9-UK/samples/_14-frac_points_1e-24/historical/*/period_1994_2014-/"
dir_cf = "outputs/outputs_scratch/UK/isimip/test-ukSPECIFIC9-UK/samples/_14-frac_points_1e-24/"

obs_file = "data/data/driving_data2526/UK/isimp3a/obsclim/GSWP3-W5E5/period_2002_2019/burned_area.nc"

obs = iris.load_cube(obs_file)

experiments = ["Evaluate", "Standard_0", "Standard_1"]
experiment_names = ['Burned Area', 'Fuel', 'Dryness']
ssps = ['ssp585', 'ssp370', 'ssp126']
periods = ["period_2090_2099-", "period_2040_2049-"]
cfyears = [[2090, 2099], [2040, 2049]]
af_cmaps = ["diverging_BlueRed", "diverging_PinkGreen", "diverging_TealOrange"]

def plot_experiment(experiment, i0, dir_f, dir_cf, ssp, period, cfyear):
    dir_f = dir_f + experiment + '/'
    dir_cf = dir_cf + ssp + '/*/' + period + '/' +experiment + '/'
    i = i0 * 6
    if i0 == 0:
        af_title = "Amplification factor 5%"
        af_title2 = "Amplification factor 95%"
        pval_title = "Likelihood"
    else:
        af_title = ""
        af_title2 = ""
        pval_title = ""
        
    plot_change_in_burned_area_future(dir_f, dir_cf,
                                      obs_file = obs_file, year = [1995, 2014],
                                      cfyear = cfyear, 
                                      af_title = af_title, af_title2 = af_title2,
                                      pval_title = pval_title,
                                      af_cmap = af_cmaps[i0], axes = axes[i:(i+3)])
    
    axes[i].text(
        -0.1, 0.5, experiment_names[i0],
        transform=axes[i].transAxes,
        rotation=90,
        va="center",
        fontweight="bold",
        ha="center"
    )

    plot_change_in_burned_area_future(dir_f, dir_cf,
                                      obs_file = obs_file, year = [1995, 2014],
                                      cfyear = cfyear, max_BA = True,
                                      af_title = af_title, af_title2 = af_title2,
                                      pval_title = pval_title,
                                      af_cmap = af_cmaps[i0], axes = axes[(i+3):(i+6)])
    #set_trace()

for ssp in ssps:
    for period, cfyear in zip(periods, cfyears):
        print(period)
        print(cfyear)
        fig, axes = set_up_sow_plot_windows(3, 6, obs,  
                                    size_scale = 2 + obs.shape[1]/obs.shape[0],         
                                    oma = [0.6, 0.9, 0.5, 0.25])
        [plot_experiment(exp, i, dir_f, dir_cf, ssp, period, cfyear) \
            for i, exp in enumerate(experiments)]
        fig.suptitle(ssp + ' ' + str(cfyear[0]) + '-' + str(cfyear[1]), y=0.99)
        plt.savefig("figs/" + ssp + period + '.png', dpi=600)
        #set_trace()
