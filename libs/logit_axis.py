
from pdb import set_trace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def scale2upper1(y):
    #set_trace()
    return y/(1.0 + y)
    #return 1-np.exp(-y * (-np.log(0.5)))

def scale2upper1_inverse(z):
    return z/(1.0 - z)
    #return -np.log(1 - z) / np.log(2)

def scale2upper1_labels(ytick_labels):
    # Compute difference from 1
    diffs = ytick_labels - 1

    # Format labels
    formatted_labels = []
    for d in diffs:
        if np.isclose(d, 0):
            formatted_labels.append("1")
        else:
            sign = "+" if d > 0 else "-"
            magnitude = abs(d)
            formatted_labels.append(f"1 {sign} {magnitude:.6f}")
    return formatted_labels

def scale2upper1_axis(ax, ytick_labels = None, ylim = None):
    ax.set_yticks([])          # remove ticks
    ax.set_yticklabels([])     # remove tick labels
    ytick_labels_txt = None
    
    if ytick_labels is None:
        if ylim is None or ylim[0] < 0.2:
            ylim = [0,1]
            ytick_labels = np.array([0, 0.2, 0.5, 1, 2, 5])
            ytick_labels_txt = np.array(['0', '1/5', '1/2', 'no\nchange\n', '2', '5', ' '])
        else:
             
            y0 = signif(1-scale2upper1_inverse(ylim[0]), 1)
            ytick_labels = np.array([-y0, -y0/2, 0, y0/2, y0]) + 1
            
            if len(ylim) == 1:
                ylim = [ylim[0], 1-ylim[0]]
            
    else:
        if ylim is None:
            ylim = np.range( ytick_labels)                                                                                   

    # Step 1: Choose locations in transformed space (display space)
    yticks_transformed = scale2upper1(ytick_labels)
    yticks_transformed = np.append(yticks_transformed, 1)
    
    # Step 2: Invert to get original y values (for labeling)
    if ytick_labels_txt is None: 
        ytick_labels_txt = [f"{v:.2f}" for v in ytick_labels] + ['']
    if len(np.unique(ytick_labels_txt)) < len(ytick_labels):
        ytick_labels_txt = scale2upper1_labels(ytick_labels) + ['']
    
    # Step 3: Apply to plot
    try:
        ax.set_yticks(yticks_transformed)
        ax.set_yticklabels(ytick_labels_txt)#, ha = 'center')#, rotation = 90
    except:
        set_trace()
    ax.set_ylim(ylim)

