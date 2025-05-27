import earthaccess
import xarray as xr
import rioxarray as rxr
import rasterio
import numpy as np
from scipy.interpolate import griddata
from pdb import set_trace
set_trace()

import os
# Set proxy for requests
os.environ["HTTP_PROXY"] = "http://iboss-proxy-3.metoffice.gov.uk:port"
os.environ["HTTPS_PROXY"] = "http://iboss-proxy-3.metoffice.gov.uk:port"

#earthaccess.login(strategy="netrc")
# ---- STEP 1: Authenticate and Download VCF Tree Cover Data ----
auth = earthaccess.login(strategy="interactive", persist=True)

# Search for MOD44B data (VCF)
granules = earthaccess.search_data(
    short_name="MOD44B",
    cloud_hosted=True, 
    bounding_box=[-180, -90, 180, 90],  # Global dataset
    temporal=("2023-01-01", "2023-12-31")  # Choose a time range
)


# Download the first granule (modify if needed)
downloaded_file = earthaccess.download(granules[0])
