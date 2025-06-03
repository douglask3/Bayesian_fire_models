import numpy as np
from pdb import set_trace
import glob
import os
from random import randrange
import iris
from iris_plus import *
from constrain_cubes_standard import *
from scipy.interpolate import RegularGridInterpolator
import datetime


def find_most_likely_cube(filename):    
    cubes = iris.load(filename)
    
    # Filter cubes that have both 'latitude' and 'longitude' coordinates
    for cube in cubes:
        coord_names = [coord.name() for coord in cube.coords()]
        if 'latitude' in coord_names and 'longitude' in coord_names:
            data_cube = cube
            break  # Stop after finding the first matching cube
    
    return(data_cube)

def read_variable_from_netcdf_from_dir(dir, filename, find_no_files = False, ens_no = None):
    print("Opening:")
    print(filename)
    print("From:")
    print(dir)
    print("At:")
    print(datetime.datetime.now())
    
    if filename[0] == '~' or filename[0] == '/' or filename[0] == '.': 
        dir = ''
    
    if find_no_files or ens_no is not None:
        files = glob.glob(dir + '**', recursive = True)
        files = [file for file in files if filename in file]
        if find_no_files:
            return(len(files))

    if filename[-3:] != '.nc': 
        filename = filename + '.nc'
    
    try:
        if ens_no is not None and len(files) > 1:
            try:    
                dataset = iris.load_cube(files[ens_no], callback=sort_time)
            except:
                dataset = iris.load_cube(files[ens_no])
        elif isinstance(filename, str):        
            dataset = iris.load_cube(dir + filename, callback=sort_time)
        else:
            dataset = iris.load_cube(dir + filename[0], filename[1], callback=sort_time)
    except:
        try:
            dataset = iris.load_cube(dir + filename)
        except:
            try:
                dataset = find_most_likely_cube(dir + filename)
                #set_trace()
                print("WARNING!: mutliple cubes in file.")
                print("\t returning first with lats and lons")
                
            except:
                dataset = None
    #if filename == "cg_strokes.nc":
    #    set_trace()
    return dataset


from cf_units import Unit
def convert_time_to_standard(time_coord, calendar = 'proleptic_gregorian'):
    # Get the origin from the existing time coord
    origin = time_coord.units.origin  # e.g. 'days since 1850-01-01'

    # Create a new Unit with the desired calendar
    new_units = Unit(origin, calendar = calendar)  # or 'proleptic_gregorian'

    # Convert numeric time to datetimes using the old unit
    datetimes = time_coord.units.num2date(time_coord.points)

    # Assign new units
    time_coord.units = new_units

    # Convert datetime back to numeric time using new calendar
    time_coord.points = new_units.date2num(datetimes)
    return time_coord

def interpolate_time(dataset, time_points):
    
    '''
    import cf_units

    # Get dataset's time units
    dataset_time = dataset.coord('time')
    
    # Get the origin date string from the original time_points
    origin = time_points.units.origin

    # Create a new Unit object with the same origin but matching the dataset's calendar
    new_units = cf_units.Unit(f"days since {origin}", calendar=dataset_time.units.calendar)

    # Apply the new units to a copy of time_points
    target_time = time_points.copy()
    target_time.units = new_units

    # Convert to dataset's units (they now have matching calendars)
    target_time.convert_units(dataset_time.units)

    # Interpolate
    dataset_interp = dataset.interpolate([('time', target_time)], iris.analysis.Linear())

    return dataset_interp
    '''
    '''
    # Get dataset's time units
    dataset_time = dataset.coord('time')

    # Make a copy of the target time points
    target_time = time_points.copy()

    # Manually set calendar to match dataset's calendar
    set_trace()
    target_time.units = target_time.units(calendar=dataset_time.units.calendar)

    # Now convert units
    target_time.convert_units(dataset_time.units)

    # Then interpolate
    dataset_interp = dataset.interpolate([('time', target_time)], iris.analysis.Linear())
    return dataset_interp
    '''
    
    # Make a copy of your target time points
    target_time = time_points.copy()
    
    if target_time.units != dataset.coord('time').units:
        if target_time.units.calendar != dataset.coord('time').units.calendar:
            target_time = convert_time_to_standard(target_time, 
                                                   dataset.coord('time').units.calendar)
        #set_trace()
        # Get the time coordinate from the dataset
        dataset_time = dataset.coord('time')
    
        # Convert target_time to the same units as dataset_time
        target_time.convert_units(dataset_time.units)
    
    
    # Now you can safely interpolate 
    dataset_interp = dataset.interpolate([('time', target_time.points)], 
                                         iris.analysis.Linear())

    return dataset_interp

def read_variable_from_netcdf(filename, dir = '', subset_function = None, 
                              make_flat = False, units = None, 
                              subset_function_args = None,
                              time_series = None, time_points = None, extent = None,
                              return_time_points = False, return_extent = False,
                              find_no_files = False,
                              ens_no = None,
                              *args, **kw):
    """Read data from a netCDF file 
        Assumes that the variables in the netcdf file all have the name "variable"
        Assunes that values < -9E9, you dont want. This could be different in some circumstances
    Arguments:
        filename -- a string with filename or two element python list containing the name of 
            the file and the target variable name. If just the sting of "filename" 
            assumes variable name is "variable"
        dir -- directory file is in. Path can be in "filename" and None means no 
            additional directory path needed.
        subset_function -- a function or list of functions to be applied to each data set
        subset_function_args -- If subset_function is a function, dict arguments for that function. 
                            If subset_function is a list, a  list or dict constaining arguments 
                                for those functions in turn.
        make_flat -- should the output variable to flattened or remain cube
        time_series -- list comtaining range of years. If making flat and returned a time series, 
            checks if that time series contains year.
    Returns:
        Y - if make_flat, a numpy vector of the target variable, otherwise returns iris cube
    """
    dir0 = dir
    dir = dir.split('||')
    i = 0
    dataset = None
    
    while i < len(dir) and dataset is None:
        dataset = read_variable_from_netcdf_from_dir(dir[i], filename, find_no_files,
                                                     ens_no = ens_no)
        #if filename == "cg_strokes.nc":
        #    set_trace()
        i += 1
        dataset00 = dataset.copy()
    if dataset is None:
        set_trace()
        print("==============\nERROR!")
        print("can't open data.")
        print("Check directory (''" + dir0 + "''), filename (''" + filename + \
              "'') or file format")
        print("==============")
    if find_no_files: return dataset
    coord_names = [coord.name() for coord in dataset.coords()]
    if time_points is not None:     
        if 'time' in coord_names:
            dataset = interpolate_time(dataset, time_points)
            #set_trace() 
            #dataset = dataset.interpolate([('time', time_points)], iris.analysis.Linear())
        else:   
            def addTime(time_point):
                time = iris.coords.DimCoord(np.array([time_point.points]), standard_name='time',
                                            units = time_points.units)
                dataset_cp = dataset.copy()
                dataset_cp.add_aux_coord(time)
                return dataset_cp

            dataset_time = [addTime(time_point) for time_point in time_points.points]
            dataset = iris.cube.CubeList(dataset_time).merge_cube()
            dataset0 = dataset.copy()
    if extent is not None:
        dataset = dataset.regrid(extent, iris.analysis.Linear())
        dataset0 = dataset.copy()
    
    if units is not None: dataset.units = units
    if subset_function is not None:
        if isinstance(subset_function, list):
            for FUN, args in zip(subset_function, subset_function_args):
                try:    
                    dataset = FUN(dataset, **args)
                except:
                    print("Warning! function: " + FUN.__name__ + " not applied to file: " + \
                          dir + filename)
        else:      
            dataset = subset_function(dataset, **subset_function_args) 
    if return_time_points: 
        time_points = dataset.coord('time')
        
    
    if return_extent:
        extent = dataset[0]
    
    if make_flat: 
        if time_series is not None: years = dataset.coord('year').points
        
        try:    
            dataset = dataset.data.flatten()
        except:
            set_trace()
            
        if time_series is not None:
            if not years[ 0] == time_series[0]:
                dataset = np.append(np.repeat(np.nan, years[ 0]-time_series[0]), dataset)
            if not years[-1] == time_series[1]:
                dataset = np.append(dataset, np.repeat(np.nan, time_series[1]-years[-1]))
            if return_time_points: set_trace()
     
    if return_time_points: dataset = (dataset, time_points)
    if return_extent:  dataset += (extent,)
    
    return dataset

def read_all_data_from_netcdf(y_filename, x_filename_list, CA_filename = None, 
                              add_1s_columne = False, 
                              y_threshold = None, x_normalise01 = False, scalers = None,
                              check_mask = True, frac_random_sample = 1.0, 
                              min_data_points_for_sample = None,
                              x_find_mode = 'single', 
                              dir_driving_data = None, *args, **kw):

                              
    """Read data from netCDF files 
        
    Arguments:
        y_filename -- a two element python list containing the name of the file and the target 
            variable name
        x_filename_list -- a python list of filename containing the feature variables
        CA_filename -- a python list of filename containing the area of the cover type
        y_threshold -- if converting y into boolean, the threshold we use to spit into 
            0's and 1's
        add_1s_columne -- useful for if using for regressions. Adds a variable of 
            just 1's t rperesent y = SUM(a_i * x_i) + c
        x_normalise01 -- Boolean. If True, then X's are normalised between 0 and 1.
        scalers -- None or numpy array of shape 2 by n. columns of X.
            Defines what scalers (min and max) to apply to each X column. 
            If None, doesn't apply anything.
        check_mask -- Boolean. If True, simple checks if there are any large negtaive numbers 
            and makes them out. Assunes that values < -9E9, you dont want. 
            This could be different in some circumstances
        frac_random_sample -- fraction of data to be returned
        see read_variable_from_netcdf comments for *arg and **kw.
    Returns:
        Y - a numpy array of the target variable
        X - an n-D numpy array of the feature variables 
    """
    
    Y, time_points, extent = read_variable_from_netcdf(y_filename, make_flat = True, *args, 
                                    return_time_points = True, return_extent = True, **kw)
    
    if CA_filename is not None:
        CA = read_variable_from_netcdf(CA_filename, make_flat = True, 
                                       time_points = time_points, extent = extent, *args, **kw)
   
    # Create a new categorical variable based on the threshold
    if y_threshold is not None:
        Y = np.where(Y >= y_threshold, 0, 1)
        #count number of 0 and 1 
        counts = np.bincount(Y)
        #print(f"Number of 0's: {counts[0]}, Number of 1's: {counts[1]}")   
    
    def open_ensemble_member(ens_no, Y, scalers, frac_random_sample,
                             cells_we_want = None):
        n=len(Y)
        m=len(x_filename_list)
    
        X = np.zeros([n,m])
        for i, filename in enumerate(x_filename_list):
            X[:, i] = read_variable_from_netcdf(filename, make_flat = True, 
                                                time_points = time_points,
                                                extent = extent, ens_no = ens_no,
                                                *args, **kw)
   
        if add_1s_columne: 
            X = np.column_stack((X, np.ones(len(X)))) # add a column of ones to X 
    
        if check_mask:
            if CA_filename is not None:
                if cells_we_want is None:
                    cells_we_want = np.array([np.all(rw > -9e9) and np.all(rw < 9e9) 
                                             for rw in np.column_stack((X, Y, CA))])
                CA = CA[cells_we_want]
            else:
                # Apply conditions separately to X and Y
                X_mask = np.all((X > -9e9) & (X < 9e9), axis=1)  # Check all columns in X
                Y_mask = (Y > -9e9) & (Y < 9e9)  # Apply directly to Y
        
                # Combine the two masks
                if cells_we_want is None: cells_we_want = X_mask & Y_mask
                
            Y = Y[cells_we_want]
            X = X[cells_we_want, :]
            
        if x_normalise01 and scalers is None: 
            try:
                scalers = np.array([np.min(X, axis=0), np.max(X, axis=0)])
            except:
                set_trace()
            squidge = (scalers[1,:]-scalers[0,:])/(X.shape[0])
            scalers[0,:] = scalers[0,:] - squidge
            scalers[1,:] = scalers[1,:] + squidge
            
            test = scalers[1,:] == scalers[0,:]
            scalers[0,test] = 0.0
            scalers[1,test] = 1.0
        
    
        if frac_random_sample is None: 
            frac_random_sample = 1000
        else:
            if min_data_points_for_sample is not None:
                min_data_frac = min_data_points_for_sample/len(Y)
                if min_data_frac > frac_random_sample: frac_random_sample = min_data_frac
    
        if frac_random_sample < 1:
            M = X.shape[0]
            selected_rows = np.random.choice(M, size = int(M * frac_random_sample), 
                                             replace=False)
            Y = Y[selected_rows]
            X = X[selected_rows, :]
            if CA_filename is not None:
                CA = CA[selected_rows]
        
        if scalers is not None:
            X = (X-scalers[0, :]) / (scalers[1, :] - scalers[0, :])
            if check_mask: 
                if CA_filename is not None: return Y, X, CA, cells_we_want, scalers
            return Y, X, cells_we_want, scalers
            
        if check_mask or frac_random_sample: 
            if CA_filename is not None:  return Y, X, CA, cells_we_want
        return Y, X, cells_we_want
        
        if CA_filename is not None: return Y, X, CA

        return Y, X

    
    if x_find_mode == 'ensemble-single':
        nfs = [read_variable_from_netcdf(filename, find_no_files = True, *args, **kw)    
               for  filename in x_filename_list]
        nfs = np.array(nfs)
        nfs = np.unique(nfs[nfs >1])
        if len(nfs) == 0:
            output = open_ensemble_member(None, Y, scalers, frac_random_sample)
        elif len(nfs) == 1:
            xOut = []
            cells_we_want = None
            for i in range(nfs[0]): 
                print(i)
                out_file = dir_driving_data + 'ens_no-' + str(i) + '.npy'
                if not os.path.isfile(out_file) or i == 0:
                    output = open_ensemble_member(i, Y, scalers, frac_random_sample,
                                                  cells_we_want = cells_we_want)
                    if i == 0: cells_we_want = output[2]
                    np.save(out_file, output[1])
                
                xOut = xOut + [out_file]
            
            y = list(output)
            y[1] = xOut
            output = tuple(y)         
        else:
            set_trace()
    else:
        output = open_ensemble_member(None, Y, scalers, frac_random_sample)
    return output
