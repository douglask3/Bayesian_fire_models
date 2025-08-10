import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator
from matplotlib.transforms import Bbox
from matplotlib import patches
from matplotlib.ticker import FuncFormatter

file_pattern_1 = r"C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/means/plotB/*.csv"
file_pattern_2 = r"C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/pc-95.0/plotB/*.csv"

csv_files_1 = glob.glob(file_pattern_1)
csv_files_2 = glob.glob(file_pattern_2)

#print(csv_files_1)
#print(csv_files_2)

#exit()

#create one subplot per file
fig, axes = plt.subplots(nrows=4,ncols=2, figsize=(6,12), sharex=True)
n_rows = axes.shape[0] #extracts the number of rows

#one color per subplot
colors = [
    ('#c7384e', '#fc6'), #burned area
    ('#e98400', '#e9840000'), #fuel
    ("#ee007f", '#ee007f00'), #human
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
##extracting data
    x = np.arange(len(stats.columns)) 
    years = 2003 + x // 12 
    months = x % 12 + 1
    mean = stats.iloc[0]
    p10 = stats.iloc[1]
    p90 = stats.iloc[2] 

##plot details
    color_line, color_fill = colors[i]
    ax = axes[i, 0]
    ax.fill_between(x, p10, p90, color=color_fill, alpha=0.3, label='10th-90th Percentile') #shading between 10-90th percentiles
    ax.plot(x, mean, color=color_line, linewidth=2) #mean line
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
    #ax.yaxis.set_minor_locator(ticker.NullLocator())
    #ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=2))
    ##Remove y-labels and ticks
    #ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    #ax.tick_params(axis='x',labelsize=7)
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis='y',labelsize=7)
    ax.tick_params(axis='x',labelsize=7)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y*100:.2f}'))
    
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
    x = np.arange(len(stats.columns)) 
    years = 2003 + x // 12 
    months = x % 12 + 1
    mean = stats.iloc[0]
    p10 = stats.iloc[1]
    p90 = stats.iloc[2]
    
##plot details
    color_line, color_fill = colors[i]
    ax = axes[i, 1]
    ax.fill_between(x, p10, p90, color=color_fill, alpha=0.3, label='10th-90th Percentile') #shading between 10-90th percentiles
    ax.plot(x, mean, label='Mean', color=color_line, linewidth=2) #mean line
    ax.xaxis.set_major_locator(ticker.MultipleLocator(36)) #tick every 3 years
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{2003 + int(val) // 12}"))
    
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
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y*100:.1f}'))
    
    ##Remove minor ticks, set 
    #ax.yaxis.set_minor_locator(ticker.NullLocator())
    #ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
    #ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    ##Remove y-labels
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
  
#create shaded rectagle in each row
plt.subplots_adjust(hspace=0.15, wspace=0)
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
    # width_left = original_left - extra_width_left

#fig.text(0.5, 0.04, 'Time (months)', ha='center', fontsize=8)  # X-axis label
middle_row = n_rows // 2 # Add y-axis label for the right column (centered vertically on the column)
axes[middle_row, 1].set_ylabel('Max. Burned Area (%) allowed by control', fontsize=8)
axes[middle_row, 1].yaxis.set_label_coords(1.1,1) # Adjust label position slightly to the right for better spacing

#legend in the left y-axis
legend_labels = [
    'Burned area',
    'Fuel',
    'Ignition',
    'Weather',
]
for i in range(n_rows):
    axes[i, 0].set_ylabel(legend_labels[i], rotation = 90, fontsize=7) #labelpad=10, fontweight='bold')
    axes[i, 0].yaxis.set_label_position("left")
   # axes[i, 0].yaxis.set_label_coords(-0.02,0.4) 
    axes[i, 0].yaxis.label.set_ha('center')
    
#title for the whole plot
fig.suptitle('Congo', fontsize=10, fontweight='bold', y=0.94)    
    
#title in each column
fig.text(0.32, 0.9, 'All region', ha='center', fontsize=10, fontweight='bold') 
fig.text(0.72, 0.9, 'Extreme burned areas', ha='center', fontsize=10, fontweight='bold')    
   
#Remove top gridline and top y-tick from plot 3b (row index 2, column index 1)
#ax_target = axes[2, 1]
#plt.sca(ax_target)
#plt.yticks(
#    ticks=[0.1, 0.01],                       # Tick positions
#    labels=["$10^{-1}$", "$10^{-2}$"]        # Tick labels
#)
    
plt.show()



