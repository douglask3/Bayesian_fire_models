import iris 
import numpy as np


def above_percentile_mean(cube, cube_assess = None, percentile = 95):
    if cube_assess is None: cube_assess = cube
    area_cube = iris.analysis.cartography.area_weights(cube_assess)
    
    # Sort the cube by fractional burnt values in descending order
    sorted_indices = np.argsort(cube_assess.data.ravel())
    sorted_cube_data = cube_assess.data.ravel()[sorted_indices]
    area_data_np = np.array(area_cube.data)
    sorted_area_data = area_data_np.ravel()[sorted_indices]

    cumulative_area = np.cumsum(sorted_area_data * sorted_cube_data)

    # Determine the total area of the grid cells
    total_area = np.nansum(sorted_area_data * sorted_cube_data)

    # Find the index where the cumulative sum exceeds the percentile threshold of the total area
    threshold_index = np.argmax(cumulative_area > (percentile/100.0) * total_area)

    # Use this index to obtain the fractional burnt value 
    # corresponding to the area-weighted percentile threshold
    threshold_value = sorted_cube_data[threshold_index]#

    mask = (cube_assess.data >= threshold_value) & (~cube_assess.data.mask)
    return np.sum(cube.data[mask] * area_data_np[mask]) / np.sum(area_data_np[mask])


