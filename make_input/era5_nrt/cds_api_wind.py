import cdsapi

dataset = "derived-era5-single-levels-daily-statistics"
request = {
    "product_type": "reanalysis",
    "variable": [
        "2m_temperature",
        "10m_wind_gust_since_previous_post_processing",
        "instantaneous_10m_wind_gust"
    ],
    "year": "2025",
    "month": [
        "01", "02", "03"
    ],
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
    "daily_statistic": "daily_maximum",
    "time_zone": "utc+00:00",
    "frequency": "1_hourly",
    "area": [43, -125, 32, -114]
}

client = cdsapi.Client()
client.retrieve(dataset, request, 'temp/api_cds_grab.zip')

