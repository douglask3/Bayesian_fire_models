
import cf_units
import iris
from pdb import set_trace


#Function to sort out the time dimension
def sort_time(cube, field, filename):
    '''
    Standardises and sorts the time coordinate in an Iris cube. 
    Ensures consistent calendar units, resets time bounds, 
    and adds useful time-based coordinates (year, month).

    Inputs:
    -------
    cube     - iris.cube.Cube
               The data cube with a time coordinate to be standardised.
    field    - Not used directly in this function, but often required 
               when this function is used as a callback during loading 
               (e.g., in iris.load_cubes).
    filename - Not used directly in this function, but can be useful 
              in error messages or logging if extended later.

    Operations:
    ---------
    - Clears any existing time bounds
    - Converts time coordinate to "days since 1661-01-01" with proleptic Gregorian calendar
    - Replaces and resets the time dimension in the cube
    - Adds 'year' and 'month' auxiliary coordinates based on the time coordinate
    - Attempts to remove the 'history' attribute if present
    
    Output:
    -------
        cube     - The modified Iris cube with cleaned and standardised time metadata
    '''
    cube.coord("time").bounds=None
    tcoord = cube.coord("time")
    tcoord.units = cf_units.Unit(tcoord.units.origin, calendar="gregorian")
    tcoord.convert_units("days since 1661-01-01 00:00:00")
    tcoord.units = cf_units.Unit(tcoord.units.origin, calendar="proleptic_gregorian")
    cube.remove_coord("time")
    cube.add_dim_coord(tcoord, 0) # might need to find this dimension
    
    try:
        iris.coord_categorisation.add_year(cube, 'time')
    except:
        pass  
    try:               
        try:
            cube.remove_coord("month")
        except:
            pass
        iris.coord_categorisation.add_month_number(cube, 'time', name='month')
    except:
        pass

    try:
        del cube.attributes["history"]
    except:
        pass
    return(cube)


def add_bounds(cube):
    """
    Add bounds to common spatial and temporal coordinates in an Iris cube.

    Inputs:
    ----------
    cube : iris.cube.Cube
        An Iris cube that may lack coordinate bounds for 'time', 'longitude', or 'latitude'.

    Returns:
    -------
    iris.cube.Cube
        The same Iris cube, with bounds guessed and added (where possible) to
        the 'time', 'longitude', and 'latitude' coordinates.

    Notes:
    -----
    - If a coordinate already has bounds, or if guessing bounds fails, the function
      silently skips it (using try/except).
    - This can be useful for preparing data for operations that require coordinate bounds
      (e.g., regridding or area averaging).
    """
    coords = ('time', 'longitude', 'latitude')
    for coord in coords:
        try: 
            cube.coord(coord).guess_bounds()
        except:
            pass            
    return(cube)


def insert_data_into_cube(x, eg_cube, mask = None):
    """ insert data into cube.
    Arguments:
        x -- np array of data that we want to insert into the cube. 
             Should have same shape of eg_cube, or same length as eg_cube or length equal to Trues in mask
	eg_cube -- The cube we want to insert data into
	mask -- Boolean array of shape or length x.
                Where True, will inster data. Defaulk of None which means True for all points 
                in eg_cube.
    Returns:
        eg_cube with data replaced by x
    """

    Pred = eg_cube.copy()
    pred = Pred.data.copy().flatten()

    if mask is None:
        pred[:] = x
    else:
        pred[mask] = x

    Pred.data = pred.reshape(Pred.data.shape)
    return(Pred)

