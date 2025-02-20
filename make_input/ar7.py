import sys
sys.path.append('../libs/')
sys.path.append('libs/')
sys.path.append('.')
sys.path.append('make_inputs/')

from isimip import *

if __name__=="__main__":
    #for_region(None, None, None, region_name = 'Global')
    
    veg_frac_file = '../fireMIPbenchmarking/data/benchmarkData/vegfrac_refLC_refCW.nc'
    
    cveg_file = '../jules_benchmarking/outputs/ESACCI-BIOMASS-L4-AGB-MERGED-0.5Degree-2010-fv3.0-ensembles-10.nc'
    fire_file = '../fireMIPbenchmarking/data/benchmarkData/ISIMIP3a_obs/GFED5_Burned_Percentage.nc'

    example_file = 'data/data/driving_data/Global/isimp3a/obsclim/GSWP3-W5E5/period_2010_2012/tas_mean.nc'
    out_dir = 'data/data/driving_data/Global/isimp3a/obsclim/GSWP3-W5E5/period_2010_2012/'

    
    frac = iris.load(veg_frac_file)[0]
    frac.coord('latitude').coord_system = None
    frac.coord('longitude').coord_system = None

    tree = (frac[[0, 1, 4]]).collapsed('type', iris.analysis.SUM)
    totalVeg = (frac[[0, 1, 2, 3, 4]]).collapsed('type', iris.analysis.SUM)
    
    cveg = iris.load(cveg_file)[0]
    cveg = cveg[1:].collapsed('z', iris.analysis.MEAN)

    
    burnt_area = iris.load_cube(fire_file)
    burnt_area.coord('latitude').coord_system = None
    burnt_area.coord('longitude').coord_system = None

    
    example = iris.load_cube(example_file)

    cveg = cveg.regrid(example, iris.analysis.Linear())
    tree = tree.regrid(example, iris.analysis.Linear())
    totalVeg = totalVeg.regrid(example, iris.analysis.Linear())
    burnt_area = burnt_area.regrid(example, iris.analysis.Linear())

    # Convert time points to datetime objects
    burnt_area_times = burnt_area.coord('time').units.num2date(burnt_area.coord('time').points)
    example_times = example.coord('time').units.num2date(example.coord('time').points)

    # Get start and end times from the example cube
    start_time = example_times.min()
    end_time = example_times.max()
    
    start_idx = np.where(burnt_area_times >= start_time)[0][0] -1
    # Find the index where burnt_area_times <= end_time
    end_idx = np.where(burnt_area_times <= end_time)[0][-1] + 1

    # Find the index where burnt_area_times <= end_time
    constrained_burnt_area = burnt_area[start_idx:end_idx]
    
    
    
    def save_ncdf(cube, varname, dir = out_dir): 
        iris.save(cube, dir + '/' + varname +  '.nc')

    save_ncdf(tree, 'tree_cover_cci')
    save_ncdf(totalVeg, 'total_veg_cover_cci')
    save_ncdf(cveg, 'cveg')
    save_ncdf(constrained_burnt_area, 'burnt_area')


    veg_frac_file = '../fireMIPbenchmarking/data/benchmarkData/veg_cont_fields_CRU.nc' 
    def make_vcf_veg_frac_ancil(varname):
        cube = iris.load_cube(veg_frac_file, varname) 
        cube.coord('lat').rename('latitude')
        cube.coord('lon').rename('longitude')
        cube.coord('latitude').coord_system = None
        cube.coord('longitude').coord_system = None
        cube.coord('latitude').units = cf_units.Unit('degrees')
        cube.coord('longitude').units = cf_units.Unit('degrees')
        cube = cube.regrid(example, iris.analysis.Linear())
        cube.data[cube.data < 0.0] = 0.0
        cube.data = cube.data/100.0
        return(cube)
    
    tree = make_vcf_veg_frac_ancil('Tree_cover')
    
    save_ncdf(tree, 'Tree_cover' + '_vcf')

    veg = make_vcf_veg_frac_ancil('bare_ground')
    bg = veg.copy()
    veg.data = 1.0 - veg.data
    veg.rename('Total_cover')
    save_ncdf(veg, 'Total_cover' + '_vcf')
    
    file_list0 = ["burnt_area.nc", "lightning.nc", "Total_cover_vcf.nc",
                 "consec_dry_mean.nc",   "pr_mean.nc", "Tree_cover_vcf.nc", 
                 "cveg.nc", "tas_max.nc", "vpd_max.nc", "dry_days.nc", 
                  "tas_mean.nc", "vpd_mean.nc"]
    cubes = [iris.load_cube(out_dir + file) for file in file_list0]
    #cubes = iris.load(file_list)
    #set_trace()
    # Start with a fully "valid" mask (all True)
    common_mask = None

    for cube in cubes:
        # Convert masked data to NaN (if not already)
        data = cube.data
        data = np.ma.filled(data, np.nan)  # Convert masked to NaN
        
        # Create a mask where NaNs exist
        if data.ndim == 3:
            mask = np.isnan(data).any(axis=0)  # Result: 2D (lat, lon)
        else:
            mask = np.isnan(data)  # Already 2D, no collapse needed
    
        # Combine with the existing common mask
        if common_mask is None:
            common_mask = mask
        else:
            common_mask = np.logical_or(common_mask, mask)  # If NaN in one, set NaN in all

    for cube in cubes:
        
        try:
            data = np.ma.array(cube.data, mask=common_mask)  # Apply the common mask
        except:
            expanded_mask = np.broadcast_to(common_mask, cube.data.shape)
            data = np.ma.array(cube.data, mask=expanded_mask)
        cube.data = data

    
    makeDir(out_dir + '/masked/')
    makeDir('data/Pantanal_example/')
    for cube, file in zip(cubes, file_list0):
        save_ncdf(cube, '/masked/' + file[0:-3])
        Pcube = constrain_natural_earth(cube, Country = 'Brazil')
        Pcube = constrain_BR_biomes(Pcube, [6])
        save_ncdf(Pcube, file[0:-3], 'data/Pantanal_example/')
        
    
