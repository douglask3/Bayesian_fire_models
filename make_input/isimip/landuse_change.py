import iris
import numpy as np



# Load the cube
def work_out_change(dir, file_in, file_out):
    cube = iris.load_cube(dir + file_in)

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

    # Update metadata
    diff_cube.standard_name = None
    diff_cube.long_name = 'change_in_' + (cube.name() or 'variable')
    diff_cube.rename('change_in_' + (cube.name() or 'variable'))

    # Save to NetCDF
    iris.save(diff_cube, dir + file_out)


if __name__=="__main__":
    dir = "data/data/driving_data2425/Amazon/isimp3a/obsclim/GSWP3-W5E5/period_2000_2019/"
    files_in = ["debiased_tree_cover_jules-es.nc","crop_jules-es.nc",  "pasture_jules-es.nc"]
    files_out = ["debiased_tree_cover_change_jules-es.nc",
                 "crop_change_jules-es.nc", "pasture_change_jules-es.nc"]

    [work_out_change(dir, file_in, file_out) for file_in, file_out in zip(files_in, files_out)]

