import iris
import numpy as np
from pdb import set_trace


# Load the cube
def work_out_change(dir, file_in, file_out):
    
    #if "counterclim" in dir and file_in == 'debiased_tree_cover_jules-es.nc':
    #    set_trace()

    if "_loss_" in file_out and "counterclim" in dir:
        filename = dir.replace('counterclim', 'obsclim') + file_in
    else:
        filename = dir + file_in
    cube = iris.load_cube(filename)

    # Check time dimension
    time_dim = cube.coord_dims('time')[0]
    data = cube.data

    # Calculate timestep-to-timestep change (forward difference)
    diff_data = np.diff(data, axis=time_dim)

    # Extrapolate backwards by repeating the first difference
    first_diff = diff_data.take(indices=0, axis=time_dim)
    extrapolated = np.expand_dims(first_diff, axis=time_dim)

    # Concatenate to match original shape
    full_diff_data = np.concatenate((extrapolated, diff_data), axis=time_dim)

    # Create a new cube
    diff_cube = cube.copy(full_diff_data)
    if "_loss_" in file_out:
        diff_cube.data = - diff_cube.data
        diff_cube.data[diff_cube.data < 0.0] = 0.0
    # Update metadata
    diff_cube.standard_name = None
    diff_cube.long_name = 'change_in_' + (cube.name() or 'variable')
    diff_cube.rename('change_in_' + (cube.name() or 'variable'))

    # Save to NetCDF
    iris.save(diff_cube, dir + file_out)


def run_LULCC_for_all_files(dir,
                            files_in  = ["debiased_tree_cover_jules-es.nc",
                                         "debiased_tree_cover_jules-es.nc",
                                           "crop_jules-es.nc", "pasture_jules-es.nc"],
                            files_out = ["debiased_tree_cover_change_jules-es.nc",
                                         "debiased_tree_cover_loss_jules-es.nc",
                                         "crop_change_jules-es.nc", 
                                         "pasture_change_jules-es.nc"]):

    return [work_out_change(dir, file_in, file_out) \
            for file_in, file_out in zip(files_in, files_out)]

def run_LULCC_for_all_regions(regions, dir_base = "data/data/driving_data_base/",
                              clims = ["obsclim", "counterclim"],
                              periods3a = ["period_2000_2019", "period_1901_1920"],
                              experiments = ["historical", "ssp370", "ssp585",  "ssp126"],
                              models = ["GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR", 
                                        "MRI-ESM2-0", "UKESM1-0-LL"],
                              periods3b = ["period_1994_2014", "period_2015_2099", 
                                           "period_2015_2099", "period_2015_2099"],
                              *args, **kw): 
    
    for region in regions:
        region = region.replace(' ', '_')
        for clim in clims:
            for period in periods3a:
                dir = dir_base + region + "/isimp3a/" + clim + \
                      "/GSWP3-W5E5/" +  period + "/"
                
                run_LULCC_for_all_files(dir, *args, **kw)

        for experiment in experiments:
            for model in models:
                for period in periods3b:
                    dir = dir_base + region + "/isimp3b/" + \
                          experiment + "/" + model + "/" + period + "/"

                    try:
                        run_LULCC_for_all_files(dir, *args, **kw)
                    except:
                        print("not run, likely no dir:" + dir)

if __name__=="__main__":
    
    regions = ['NEIndia', 'Alberta', 'LA', 'Congo','Pantanal', 'Amazon']
    regions = ['Pantanal', 'Amazon']
    run_LULCC_for_all_regions(regions)
    

