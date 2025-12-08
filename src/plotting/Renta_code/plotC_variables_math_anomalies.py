import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator
from matplotlib.transforms import Bbox
from matplotlib import patches
from matplotlib.ticker import FuncFormatter

def process_anomalies(last_18_pattern, clim_pattern):
    # Get lists of files
    last_18_files = glob.glob(last_18_pattern)
    clim_files = glob.glob(clim_pattern)
    # Build dict for quick matching by base prefix (remove suffix)
    def base_prefix(f, suffix):
        return os.path.basename(f).replace(suffix, '')
    
    last_18_dict = {base_prefix(f, '_copy_last_18.csv'): f for f in last_18_files}
    clim_dict = {base_prefix(f, '_copy_monthly_climatology.csv'): f for f in clim_files}
    
     # For each climatology file, find matching last_18 file
    for base in clim_dict:
        if base in last_18_dict:
            clim_file = clim_dict[base]
            last_18_file = last_18_dict[base]
            
            df_clim = pd.read_csv(clim_file, header=None)
            df_last_18 = pd.read_csv(last_18_file, header=None)
            
            # Print shape info for debug
            #print(f"Checking files: {base}")
            #print(f"  {os.path.basename(clim_file)} shape: {df_clim.shape}")
            #print(f"  {os.path.basename(last_18_file)} shape: {df_last_18.shape}")
            #exit()
            
            df_clim = df_clim.apply(pd.to_numeric, errors='coerce')
            df_last_18 = df_last_18.apply(pd.to_numeric, errors='coerce')
            
            df_anomaly = df_last_18.values - df_clim.values
            
            anomaly_path = clim_file.replace('_copy_monthly_climatology.csv', '_copy_anomaly.csv')
            pd.DataFrame(df_anomaly).to_csv(anomaly_path, index=False, header=False)
                       
df_1 = r"C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/means/plotC/*_copy_last_18.csv"
df_2 = r"C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/pc-95.0/plotC/*_copy_last_18.csv"
df_1_clim = r"C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/means/plotC/*_copy_monthly_climatology.csv"
df_2_clim = r"C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/pc-95.0/plotC/*_copy_monthly_climatology.csv"

process_anomalies(df_1,df_1_clim)
process_anomalies(df_2,df_2_clim)