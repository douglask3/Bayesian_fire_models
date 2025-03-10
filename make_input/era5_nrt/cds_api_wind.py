import cdsapi
from pdb import set_trace
import os.path
import zipfile


def download_era5(year = 1940, months = range(13), area = [90, -180, -90, 180],
              dataset = "derived-era5-single-levels-daily-statistics", temp_dir = 'temp/'):

    months = ['0' + str(i) if i < 10 else str(i) for i in months]
   
    
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
        set_trace()
        #with zipfile.ZipFile(file, 'r') as zip_ref:
        #    zip_ref.extractall(file[:-4])
        set_trace()
            
    
    process_var("2m_temperature", "daily_maximum")
    process_var("10m_wind_gust_since_previous_post_processing", "daily_maximum")
    process_var("instantaneous_10m_wind_gust", "daily_maximum")
    process_var("2m_temperature", "daily_maximum")
    process_var("2m_dewpoint_temperature", "daily_minimum")
    process_var("volumetric_soil_water_layer_1", "daily_minimum")

    #request["variable"] = [
    #        "2m_dewpoint_temperature",
    #        "volumetric_soil_water_layer_1"
    #    ],
    #request["daily_statistic"] = "daily_minimum"
    

    #client = cdsapi.Client()
    #client.retrieve(dataset, request).download()
    #client.retrieve(dataset, request, temp_dir + '/daily_minimum.zip')


if __name__=="__main__":
    area = [36, -121, 32, -114]
    temp_dir = "/scratch/dkelley/Bayesian_fire_models/temp/era5_nrt/"
    download = True
    
    if download: download_era5(months = range(13), area = area, temp_dir = temp_dir)
