import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator
from matplotlib.transforms import Bbox
from matplotlib import patches
from matplotlib.ticker import FuncFormatter

file_pattern_1 = r"C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/data_csv/means/plotA/points-*_copy.csv"
file_pattern_2 = r"C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/data_csv/pc-95.0/plotA/points-*_copy.csv"

csv_files_1 = glob.glob(file_pattern_1)
csv_files_2 = glob.glob(file_pattern_2)

#print(csv_files_1)
#print(csv_files_2)

#create one subplot per file
fig, axes = plt.subplots(nrows=6,ncols=2, figsize=(6,12), sharex=True)
n_rows = axes.shape[0] #extracts the number of rows

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
    ax = axes[i, 0]
    ax.fill_between(x, p10, p90, color='gray', alpha=0.3, label='10th-90th Percentile') #shading between 10-90th percentiles
    ax.plot(x, mean, label='Mean', color='blue', linewidth=2) #mean line
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
    ##Remove minor ticks and add 2 major ticks
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
    ##Remove y-labels and ticks
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    ax.tick_params(axis='x',labelsize=7)
    
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
    ax = axes[i, 1]
    ax.fill_between(x, p10, p90, color='gray', alpha=0.3, label='10th-90th Percentile') #shading between 10-90th percentiles
    ax.plot(x, mean, label='Mean', color='blue', linewidth=2) #mean line
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
    
    ##Remove minor ticks, set 
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
    #ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    ##Remove y-labels and ticks
    #ax.tick_params(axis='y', which='both', left=False, labelleft=False)
  
#create shaded rectagle in each row
plt.subplots_adjust(hspace=0, wspace=0)
extra_width_left = 0.09
    
for row in range(n_rows):
    ax_left = axes[row, 0]
    ax_right = axes[row, 1]

    # Get bounding boxes of the left and right axes in figure coordinates
    bbox_left = ax_left.get_position()
    bbox_right = ax_right.get_position()

    # Original dimensions
    original_left = bbox_left.x0
    original_right = bbox_right.x1
    original_width = original_right - original_left
    height = bbox_left.y1 - bbox_left.y0
    bottom = bbox_left.y0

    # Expand width from left column
    width_left = original_left - extra_width_left

    # Add rectangle using figure coordinates
    rect = patches.Rectangle(
        (width_left, bottom),
        original_right,
        height,
        transform=fig.transFigure,
        facecolor='lightgray',
        edgecolor='black',
        linewidth=1.0,
        alpha=0.2,
        zorder=0
    )
    fig.add_artist(rect)

fig.text(0.5, 0.04, 'Time (months)', ha='center', fontsize=8)  # X-axis label
middle_row = n_rows // 2 # Add y-axis label for the right column (centered vertically on the column)
axes[middle_row, 1].set_ylabel('Ensemble members (log scale)', fontsize=8)
axes[middle_row, 1].yaxis.set_label_coords(1.1, 1.0) # Adjust label position slightly to the right for better spacing

#legend in the left y-axis
legend_labels = [
    'Burned area',
    'Fragmentation',
    'Fuel',
    'Ignition',
    'Moisture',
    'Suppression',
]
for i in range(n_rows):
    axes[i, 0].set_ylabel(legend_labels[i], rotation = 360, fontsize=8, labelpad=10, fontweight='bold')
    axes[i, 0].yaxis.set_label_position("left")
    axes[i, 0].yaxis.set_label_coords(-0.02,0.4) 
    axes[i, 0].yaxis.label.set_ha('right')
    
#title in each column
fig.text(0.32, 0.9, 'Means', ha='center', fontsize=10, fontweight='bold') 
fig.text(0.72, 0.9, 'PC 95.0', ha='center', fontsize=10, fontweight='bold')    
    
plt.show()



