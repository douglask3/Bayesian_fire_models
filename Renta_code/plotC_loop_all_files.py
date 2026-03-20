import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator
from matplotlib.transforms import Bbox
from matplotlib import patches
from matplotlib.ticker import FuncFormatter

file_pattern_1 = r"C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Amazon/new_data_csv/means/plotC/*anomaly.csv"
file_pattern_2 = r"C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Amazon/new_data_csv/pc-95.0/plotC/*anomaly.csv"

csv_files_1 = glob.glob(file_pattern_1)
csv_files_2 = glob.glob(file_pattern_2)

#print(csv_files_1)
#print(csv_files_2)

#create one subplot per file
fig, axes = plt.subplots(nrows=7,ncols=2, figsize=(6,12), sharex=True)
n_rows = axes.shape[0] #extracts the number of rows

colors = [
    ('#c7384e', '#fc6'), #burned area
    ('#e98400', '#e9840000'), #fuel ##00 at the end make it transparent
    ('#e98400', '#e9840000'), #moisture
    ("#ee007f", '#ee007f00'), #ignition
    ('#ee007f', '#ee007f00'), #suppression
    ('#0096a1', '#0096a100'), #weather
    ('#0096a1', '#0096a100') #wind 
]

#loop through files means
for i, file_path in enumerate(csv_files_1):
    print(f"Processing {file_path}")
    df = pd.read_csv(file_path, header=None)
# Convert all columns to numeric
    df = df.apply(pd.to_numeric,errors='coerce')
# Calculate stats
    mean_row = df.quantile(0.5)
    p10_row = df.quantile(0.1)
    p90_row = df.quantile(0.9)
#make sure stats are always at lines 1001, 1002, 1003 (index 1000-1002; in the case I run the code again n times)
    df.loc[1000] = mean_row
    df.loc[1001] = p10_row
    df.loc[1002] = p90_row
#overwrite files    
    df.to_csv(file_path, index=False, header=False)
#exit()
#select last 3 rows
    stats = df.tail(3)
##extracting data
    x = stats.columns
    mean = stats.iloc[0]
    p10 = stats.iloc[1]
    p90 = stats.iloc[2] 
    
##plot details
    color_line, color_fill = colors[i]
    ax = axes[i, 0]
    ax.fill_between(x, p10, p90, color=color_fill, alpha=0.3, label='10th-90th Percentile') #shading between 10-90th percentiles
    ax.plot(x, mean, label='Mean', color=color_line, linewidth=2) #mean line
    month_labels = ['Sep 2023', 'Oct', 'Nov', 'Dec', 'Jan 2024', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec' , 'Jan 2025', 'Feb'] * (len(x) // 12)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(month_labels, rotation=45, ha='right', fontsize=7)
    ##log transformation on y-axis
   #ax.set_yscale('log')
    ##x-ticks grid
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    ## Remove x ticks and labels for all but last subplot in column
    if i != n_rows - 1:
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    else:
        ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
              
    ax.yaxis.set_label_position("left")
    ax.tick_params(axis='y',labelsize=7)
    ax.tick_params(axis='x',labelsize=7)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y*3592468.44:.1f}'))
    
 #loop through files pc-95.0
for i, file_path in enumerate(csv_files_2):
    print(f"Processing {file_path}")
    df = pd.read_csv(file_path, header=None)
# Convert all columns to numeric
    df = df.apply(pd.to_numeric,errors='coerce')
# Calculate stats
    mean_row = df.quantile(0.5)
    p10_row = df.quantile(0.1)
    p90_row = df.quantile(0.9)
#make sure stats are always at lines 1001, 1002, 1003 (index 1000-1002; in the case I run the code again n times)
    df.loc[1000] = mean_row
    df.loc[1001] = p10_row
    df.loc[1002] = p90_row
#overwrite files    
    df.to_csv(file_path, index=False, header=False)
#exit()
#select last 3 rows
    stats = df.tail(3)
##extracting data
    x = stats.columns
    mean = stats.iloc[0]
    p10 = stats.iloc[1]
    p90 = stats.iloc[2] 
##plot details
    color_line, color_fill = colors[i]
    ax = axes[i, 1]
    ax.fill_between(x, p10, p90, color=color_fill, alpha=0.3, label='10th-90th Percentile') #shading between 10-90th percentiles
    ax.plot(x, mean, color=color_line, linewidth=2) #mean line
    month_labels = ['Sep 2023', 'Oct', 'Nov', 'Dec', 'Jan 2024', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec' , 'Jan 2025', 'Feb'] * (len(x) // 12)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(month_labels, rotation=45, ha='right', fontsize=7)
    ##log transformation on y-axis
    #ax.set_yscale('log')
    ##x-ticks grid
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ## Remove x ticks and labels for all but last subplot in column
    if i != n_rows - 1:
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    else:
        ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
    ##Move ticks and labels to the right, adjust alignment, fontsize etc
    ax.yaxis.tick_right()             
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis='y',labelsize=7)
    ax.tick_params(axis='x',labelsize=7)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y*100:.3f}'))
  
#create shaded rectagle in each row
plt.subplots_adjust(hspace=0.15, wspace=0)

middle_row = n_rows // 2 # Add y-axis label for the right column (centered vertically on the column)
axes[middle_row, 1].set_ylabel('Max. Burned Area (%) allowed by control', fontsize=8)
axes[middle_row, 1].yaxis.set_label_coords(1.16, 1.0) # Adjust label position slightly to the right for better spacing

#legend in the left y-axis
legend_labels = [
    'Burned area',
    'Fuel',
    'Moisture',
    'Ignition',
    'Suppression',
    'Weather',
    'Wind'
]
for i in range(n_rows):
    axes[i, 0].set_ylabel(legend_labels[i], rotation = 90, fontsize=7)
    axes[i, 0].yaxis.set_label_position("left")
    axes[i, 0].yaxis.label.set_ha('center')
    axes[i, 0].yaxis.set_label_coords(-0.15,0.5) 
    
fig.text(0.02, 0.5, 'Control Contribution (km$^2$)', fontsize=8, rotation='vertical', va='center')
    
#title for the whole plot
fig.suptitle('Amazon', fontsize=12, fontweight='bold', y=0.94)    

#title in each column
fig.text(0.32, 0.9, 'All region', ha='center', fontsize=10, fontweight='bold') 
fig.text(0.72, 0.9, 'Extreme burned areas', ha='center', fontsize=10, fontweight='bold')    

#plt.show()

#save as pdf landscape orientation
fig.set_size_inches(11.7, 8.3)
plt.savefig("C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Amazon/plotC/plotC_Amazon_km2_v2.pdf", format='pdf', bbox_inches='tight')



