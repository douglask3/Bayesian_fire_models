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

sys.path.append('../../libs/')
sys.path.append('libs/')
from constrain_cubes_standard import *

from shapely.ops import unary_union
import shapely.vectorized

def download_era5(year = 1940, months = range(13), area = [90, -180, -90, 180],
              dataset = "derived-era5-single-levels-daily-statistics", temp_dir = 'temp/',
              shapefile_path = None):

    months = ['0' + str(i) if i < 10 else str(i) for i in months]
   
    if shapefile_path is not None: 
        shapes = gpd.read_file(shapefile_path)
    def download_var(variable, statistics): 
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

    def process_var(variable, statistics):
        file = download_var(variable, statistics)

        cube = iris.load_cube(file)
        #lons, lats = cube.coord('longitude').points, cube.coord('latitude').points
        lons, lats = np.meshgrid(cube.coord('longitude').points, cube.coord('latitude').points)
    
        ## Extract the shape containing "Los Angeles" in the name
        region_shape = shapes[shapes['name'].str.contains("Los Angeles", case=False, na=False)]

        # Convert to a single geometry (union of multiple polygons if needed)
        region_geom = unary_union(region_shape.geometry)


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
        cropped_cube = cube.extract(iris.Constraint(latitude=lat_constraint, longitude=lon_constraint))

        return cropped_cube            
    
    process_var("2m_temperature", "daily_maximum")
    set_trace()
    process_var("10m_wind_gust_since_previous_post_processing", "daily_maximum")
    process_var("instantaneous_10m_wind_gust", "daily_maximum")
    process_var("2m_temperature", "daily_maximum")
    process_var("2m_dewpoint_temperature", "daily_minimum")
    process_var("volumetric_soil_water_layer_1", "daily_minimum")


if __name__=="__main__":
    area = [36, -121, 32, -114]
    temp_dir = "/scratch/dkelley/Bayesian_fire_models/temp/era5_nrt/"
    shapefile_path = "data/data/SoW2425_shapes/SoW2425_Focal_MASTER_20250221.shp"
    
    download_era5(months = range(13), area = area, temp_dir = temp_dir,
                               shapefile_path = shapefile_path)
