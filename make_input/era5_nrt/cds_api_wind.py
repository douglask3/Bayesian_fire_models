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

def download_era5(years = [1940], months = range(13), area = [90, -180, -90, 180],
                  region_name = " ",
                  dataset = "derived-era5-single-levels-daily-statistics", temp_dir = 'temp/',
                  shapefile_path = None):

    months = ['0' + str(i) if i < 10 else str(i) for i in months]
   
    if shapefile_path is not None: 
        shapes = gpd.read_file(shapefile_path)

    ## Extract the shape containing region_name in the name
    region_shape = shapes[shapes['name'].str.contains(region_name, case=False, na=False)]

    # Convert to a single geometry (union of multiple polygons if needed)
    region_geom = unary_union(region_shape.geometry)

    def download_var(variable, statistics, year): 
        temp_file =  temp_dir + '/download_era5_' + variable + statistics + \
                    '_months' + '-'.join(months) + '_year' +  str(year) + '.nc'
        
        
        if os.path.isfile(temp_file): return(temp_file)
        
        request = {
            "product_type": "reanalysis",
            "variable": [variable
                #"2m_temperature",
            #"10m_wind_gust_since_previous_post_processing",
            #"instantaneous_10m_wind_gust"
            ],
            # Load the shapefile

            "year": str(year),
            "month": months,
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
        client.retrieve(dataset, request,temp_file)
        return(temp_file)

    def crop_cube(file):
        cube = iris.load_cube(file)
        #lons, lats = cube.coord('longitude').points, cube.coord('latitude').points
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
         
    def process_var(variable, statistics):
        def for_year(year):
            file = download_var(variable, statistics, year)
            return crop_cube(file)
        cubes = [for_year(year) for year in years]
        cubes = common_time_coord(cubes)
        iris.util.equalise_attributes(cubes)
        cubes = iris.cube.CubeList(cubes).concatenate_cube()
        set_trace()
        
    process_var("2m_temperature", "daily_maximum")
    set_trace()
    process_var("10m_wind_gust_since_previous_post_processing", "daily_maximum")
    process_var("instantaneous_10m_wind_gust", "daily_maximum")
    process_var("2m_temperature", "daily_maximum")
    process_var("2m_dewpoint_temperature", "daily_minimum")
    process_var("volumetric_soil_water_layer_1", "daily_minimum")


if __name__=="__main__":
    now = DT.now().year
    years = range(1940, now)
    years = range(2021, now)

    area = [36, -121, 32, -114]
    temp_dir = "/scratch/dkelley/Bayesian_fire_models/temp/era5_nrt/"
    shapefile_path = "data/data/SoW2425_shapes/SoW2425_Focal_MASTER_20250221.shp"
    region_name = "Los Angeles"
    download_era5(years, months = range(13), area = area, region_name = region_name, 
                  temp_dir = temp_dir,
                  shapefile_path = shapefile_path)
