import os.path
import pandas as pd
import pymc  as pm

import numpy as np
import iris
import iris.analysis
import iris.coord_categorisation
from pdb import set_trace


import iris.quickplot as qplt
import matplotlib.pyplot as plt



def make_ENSO_cf(filename, variable_out, time_coord = "time", monthly = True):
    cube = iris.load_cube(filename)
    
    try:
        iris.coord_categorisation.add_day_of_year(cube, time_coord, name="day_of_year")
    except:
        pass
    try:
        iris.coord_categorisation.add_month_number(cube, time_coord, name="month_number")
    except:
        pass
    try:
        iris.coord_categorisation.add_year(cube, time_coord, name='year')
    except:
        pass
    
    
    coord_clim  = 'month_number' if monthly else 'day_of_year'
    year_length = 12 if monthly else 365
    climatology = cube.aggregated_by(coord_clim, iris.analysis.MEAN)
    
    # Fallback method if direct subtraction fails due to coordinate mismatches
    clim_data = climatology.data
    # Match each time step's day_of_year (1-indexed) to the climatology array (0-indexed)
    doy_indices = cube.coord(coord_clim).points - 1
    
    anomaly = cube.copy(data=cube.data / clim_data[doy_indices])
    anomaly.rename(f"{cube.name()}_anomaly")
    

    pi_frac_yr = 2*np.pi*cube.coord(coord_clim).points/year_length
    comps = np.array([np.ones(len(pi_frac_yr)),
                      np.sin(pi_frac_yr),
                      np.sin(1*np.pi/3 + pi_frac_yr),
                      np.sin(2*np.pi/3 + pi_frac_yr),
                      np.sin(3*np.pi/3 + pi_frac_yr),
                      np.sin(4*np.pi/3 + pi_frac_yr),
                      np.sin(5*np.pi/3 + pi_frac_yr)]).T
    
    enso = pd.read_csv('data/enso_ONI.csv', sep=',', header=None)
    dates = enso.values[:,0]
    
    yrs = [int(date[-4:]) for date in dates] 
    mnths = [int(date[-7:-5]) for date in dates] 
    cYr = cube.coord('year').points
    cMn = cube.coord('month_number').points
    
    def get_time_index (Yr, Mn):
        return np.where(np.all(np.array([np.array(yrs) == Yr, np.array(mnths) == Mn]), \
                        axis = 0))[0][0]
    time_index = np.array([get_time_index(Yr, Mn) for Yr,Mn in zip(cYr, cMn)])
    ENSO = enso.values[time_index, 1]

    comps = np.asarray(comps, dtype="float64")          # Shape: (N, ncomps)
    ENSO = np.asarray(ENSO, dtype="float64")          # Shape: (N,)
    
    comps_x_index = comps * ENSO[:, None]          # Shape: (N, ncomps)
    ncomps = comps.shape[1]
    
    #result = np.apply_along_axis(
    #    esno_remove_grid_cell,
    #    axis=0,  # z dimension for shape (x, y, z)
    #    arr=cube.data,
    #    index=ENSO,
    #    comps=comps,
    #    comps_x_index = comps_x_index,
    #    ncomps = ncomps,
    #)
    
    results = [[esno_remove_grid_cell(i, j, anomaly, filename, variable_out,
                                      ENSO, comps, comps_x_index, ncomps) \
                for i in range(cube.shape[1])] for j in range(cube.shape[2])]
    set_trace()


def esno_remove_grid_cell(i, j, cube, filename, varname, index, comps, comps_x_index, ncomps,  
                          nsamples = 250, ncores = 4):
    #values_arr = np.asarray(values, dtype="float64")        # Shape: (N,)
    #def correction(C0s, Ks, comps, index):
    #    #def correction_comp(i):
    #    #    return comps[:,i] * (C0s[i] + Ks[i] * index)
    #    #icomps = [correction_comp(i) for i in range(comps.shape[1])]
    #    terms = comps_arr * (C0s + Ks * index_arr[:, None])
    #    return terms.sum(axis=1)
    #    prediction =  icomps[0] + icomps[1] + icomps[2]
    #    #return tt.sum(comps * ((C0s * Ks)[None, :] * index), axis=0)

    
    tempfile = "temp2/enso_regression/" + filename.replace('/', '_').replace('.nc', '') \
                + '-' + varname + '-' + str(i) + '-' + str(j) + '-' \
                + str(nsamples) + '-' + str(ncores) + '.csv'
    
    if os.path.isfile(tempfile): 
        out = np.loadtxt(tempfile, delimiter=",")
    
    values = cube.data[:,i,j]
    
    #values = cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN).data
    if np.sum(~values.mask) < 5 or np.sum(values.data <9E9) < 5 or \
         (~np.isnan(values)).sum() < 5:
        out = np.empty((nsamples*ncores,ncomps,)) 
        out[:] = np.nan
        np.savetxt(tempfile, out, delimiter=",")
        
        return out
    values = np.asarray(values, dtype="float64")
    set_trace()
    values = cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN).data
    with pm.Model() as for_model:
        C0s = pm.Normal('C0s', mu=0, sigma=1, shape=ncomps)
        Ks = pm.Normal('Ks', mu = 0, sigma=1, shape=ncomps)
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        #prediction = correction(C0s, Ks, comps, index)
        prediction = pm.math.dot(comps, C0s) + pm.math.dot(comps_x_index, Ks)
        if len(np.unique(values)) == 2:
            prediction = pm.math.sigmoid(prediction)
            likelihood = pm.Bernoulli("y", p=prediction, observed=values)
        else:
            likelihood = pm.Normal("y", mu=prediction, sigma=sigma, observed=values)
        
        trace = pm.sample(nsamples, return_inferencedata=True, chains = ncores)
    out = trace.posterior['Ks'].values.reshape(-1, trace.posterior['Ks'].values.shape[-1])
    set_trace()
    np.savetxt(tempfile, out, delimiter=",")
    return out


if __name__=="__main__":
    file = "data/data/driving_data_base/Amazon/burnt_area.nc"
    make_ENSO_cf(file, 'burned_area')
