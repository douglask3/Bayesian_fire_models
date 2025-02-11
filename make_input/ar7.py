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
    
    
    
    def save_ncdf(cube, varname): 
        iris.save(cube, out_dir + '/' + varname +  '.nc')

    save_ncdf(tree, 'tree_cover')
    save_ncdf(totalVeg, 'total_veg_cover')
    save_ncdf(cveg, 'cveg')
    save_ncdf(constrained_burnt_area, 'burnt_area')

    set_trace()
