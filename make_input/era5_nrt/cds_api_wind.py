import cdsapi
from pdb import set_trace
import os.path
import zipfile
import iris
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
import sys
from datetime import datetime as DT

sys.path.append('../../libs/')
sys.path.append('libs/')
from constrain_cubes_standard import *

from shapely.ops import unary_union
import shapely.vectorized

import iris
import iris.analysis
import cftime
import cf_units

def common_time_coord(cubes, time_coord = 'valid_time'):
    # Define a common reference time unit (e.g., days since 1970-01-01)
    common_time_unit = cf_units.Unit('days since 1970-01-01 00:00:00', calendar='gregorian')
    
    # Standardize time coordinates across all cubes
    for cube in cubes:
        time_coord = cube.coord('valid_time')

        # Convert numeric time to datetime
        datetimes = [time_coord.units.num2date(t) for t in time_coord.points]
        
        # Convert to a consistent cftime datetime type (Gregorian)
        new_datetimes = [cftime.DatetimeGregorian(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second) for dt in datetimes]
        
        # Convert datetime objects back to numeric time values in the common reference
        new_time_points = [common_time_unit.date2num(dt) for dt in new_datetimes]
        
        # Update the time coordinate
        time_coord.points = new_time_points
        time_coord.units = common_time_unit  # Ensure consistent units across all cubes
    
    return(cubes)

def download_era5(variables, years = [1940], months = range(13), 
                  yr_now = None, mnth_now = None, area = [90, -180, -90, 180],
                  region_name = " ",
                  dataset = "derived-era5-single-levels-daily-statistics", 
                  out_dir = 'data/',
                  temp_dir = 'temp/',
                  shapefile_path = None):
   
    if shapefile_path is not None: 
        shapes = gpd.read_file(shapefile_path)

    ## Extract the shape containing region_name in the name
    region_shape = shapes[shapes['name'].str.contains(region_name, case=False, na=False)]
    region_shape["geometry"] = region_shape["geometry"].buffer(0)
    # Convert to a single geometry (union of multiple polygons if needed)
    region_geom = unary_union(region_shape.geometry)
    
    def download_var(variable, statistics, year, mnths): 
        mnths = ['0' + str(i) if i < 10 else str(i) for i in mnths]
        temp_file =  temp_dir + '/download_era5_' + variable + statistics + \
                    '_extent_' +  '_'.join([str(ar) for ar in area]) + \
                    '_months' + '-'.join(mnths) + '_year' +  str(year) #
        if dataset != "derived-era5-single-levels-daily-statistics":
            temp_file = temp_file + dataset

        temp_file = temp_file + '.nc'
    
        print("=======")
        print("downloading")  
        print("year:" + str(year))
        print("months:" + str(mnths))
        print("variable:" +    variable)
        print("statistic:" + statistics)
        if os.path.isfile(temp_file): return(temp_file)
        
        request = {
            "product_type": "reanalysis",
            "variable": [variable
            ],
            "year": str(year),
            "month": mnths,
            "day": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12",
                "13", "14", "15",
                "16", "17", "18",
                "19", "20", "21",
                "22", "23", "24",
                "25", "26", "27",
                "28", "29", "30",
                "31"
                ],
            "daily_statistic": statistics,
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": area
        }

        client = cdsapi.Client()
        client.retrieve(dataset, request, temp_file)
        return(temp_file)

    def crop_cube(file):
        cube = iris.load_cube(file)
        lons, lats = np.meshgrid(cube.coord('longitude').points, cube.coord('latitude').points)
    
        # Create a mask where True means outside and False means inside the shape
        mask = ~shapely.vectorized.contains(region_geom, lons, lats)
        cube.data = np.where(mask, np.nan, cube.data)

        # Get bounding box (minx, miny, maxx, maxy)
        min_lon, min_lat, max_lon, max_lat = region_geom.bounds

        # Define constraint functions for lat/lon bounds
        def lon_constraint(cell):
            return min_lon <= cell <= max_lon

        def lat_constraint(cell):
            return min_lat <= cell <= max_lat

        # Apply constraints to crop the cube
        cropped_cube = cube.extract(iris.Constraint(latitude=lat_constraint, 
                                                    longitude=lon_constraint))

        return cropped_cube   
         
    def process_var(variable, statistics, variable_out):
        out_file = out_dir + '/' + region_name.replace(' ', '_')
        try:
            os.mkdir(out_file)
        except:
            pass
        
        #out_file =  out_file + '/' + dataset + '/' + variable_out + '/' + statistics + \
        #           '_years' +  str(years[0]) + '-' + str(years[-1]) 
        out_file = out_dir + '/' + region_name.replace(' ', '_') + \
                       '/Era5_' + dataset + \
                       '/' + variable_out + '/_years' +  str(years[0]) + '-' + str(years[-1]) 
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        if (yr_now is not None and years[-1] == yr_now):
            out_file = out_file + str(mnth_now)
         
        out_file = out_file + '.nc'
        
        if os.path.isfile(out_file): return(out_file)
        
        def for_month(variable, statistics, year, month):
            if year == yr_now and month > mnth_now:
                return 
            file = download_var(variable, statistics, year, [month + 1])
            return crop_cube(file)
            
        
        def for_year(year):
            cubes = [for_month(variable, statistics, year, month) for month in months]
            return cubes

        cubes = [for_year(year) for year in years]
        cubes = [x for xs in cubes for x in xs if x is not None]
        
        cubes = common_time_coord(cubes)
        iris.util.equalise_attributes(cubes)
        cubes = iris.cube.CubeList(cubes).concatenate_cube()
        iris.save(cubes, out_file)
        
        return out_file
        
    for var in variables:
        process_var(var[0], var[1], var[2])


if __name__=="__main__":
    yr_now = DT.now().year
    yearss = [range(yr_now-2, yr_now + 1), range(2010, 2026), range(2000, 2026)]
    mnth_now = DT.now().month - 2
    #day_now = DT.now().day-5
    #if day_now < 1:
    #    mnth_now = mnth_now - 1
    #    day_now = day_now + 28
    
    #years = range(1985, now + 1)
    #years = range(2000, 2026)
    dataset = "derived-era5-single-levels-daily-statistics"
    
    area = [90, -180, -60, 180]
    temp_dir = "/data/scratch/douglas.kelley/Bayesian_fire_models/temp/era5_nrt/"
    out_dir = "data/data/driving_data2425/nrt_attribution//"
    shapefile_path = "data/data/SoW2425_shapes/SoW2425_Focal_MASTER_20250221.shp"
    region_names = ["northeast India",
                   "Alberta",
                   "Los Angeles",
                   "Congo basin",
                   "Amazon and Rio Negro rivers",
                   "Pantanal basin"]
    variables = [["volumetric_soil_water_layer_1", "daily_minimum", "mrsos"],
                 ["total_precipitation", "daily_mean", "pr"], 
                 ["2m_temperature", "daily_maximum", "tasmax"],
                 ["2m_temperature", "daily_mean", "tasmin"],
                 ["2m_dewpoint_temperature", "daily_minimum", "tasdew"],
                 ["2m_temperature", "daily_minimum", "tasmin"],
                 ["10m_wind_gust_since_previous_post_processing", "daily_maximum", "WindGust1"],
                 ["instantaneous_10m_wind_gust", "daily_maximum", "WindGust2"],
                 ["evaporation", "daily_mean", "evap"],
                 ["potential_evaporation", "daily_mean", "pevap"],
                 ["runoff", "daily_mean", "mrros"]
                 ]
    
    for region_name in region_names:
        for years in yearss:
            download_era5(variables, years, months = range(12), 
                          yr_now = yr_now, mnth_now = mnth_now,
                          area = area, region_name = region_name,
                          dataset = dataset, 
                          out_dir = out_dir, 
                          temp_dir = temp_dir,
                          shapefile_path = shapefile_path)


