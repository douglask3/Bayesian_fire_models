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

df_1 = r"C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/means/plotA/*_copy.csv"
df_2 = r"C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/pc-95.0/plotA/*_copy.csv"

csv_files_1 = glob.glob(df_1)
csv_files_2 = glob.glob(df_2)
#csv_files_1_climatology = glob.glob(df_1_clim)
#csv_files_2_climatology = glob.glob(df_2_clim)

#select last 18 columns 
def get_last_18_columns(file_list):
    for file in file_list:
        df = pd.read_csv(file, header=None, skipfooter=3, engine='python')
        last_18 = df.iloc[:, -18:] #select last 18 columns
    #construct output file path
        output_path = file.replace("plotA", "plotC")
        base, ext = os.path.splitext(output_path)
        new_filename = f"{base}_last_18{ext}"
    #save
        last_18.to_csv(new_filename, index=False, header=False)

#run function
get_last_18_columns(csv_files_1)
get_last_18_columns(csv_files_2)
