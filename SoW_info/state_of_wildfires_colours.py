from matplotlib.colors import LinearSegmentedColormap
from pdb import set_trace


# Define a set of custom color gradients for various themes used in the State of Wildfires (SoW) plots.
# Each key in the dictionary represents a theme, and the corresponding value is a list of hex color codes.

SoW_cmap = {
    "gradient_hues": ["#cfe9ff", "#fc6", "#f68373", "#c7384e", "#862976"],
    
    "gradient_hues_extended": [  # Extended version of gradient_hues with smoother transitions
        "#cfe9ff",   # very light blue
        "#fbc585",   # lighter orange
        "#fc6",      # original orange
        "#f9a17e",   # orange-salmon blend
        "#f68373",   # salmon
        "#da4f58",   # red-orange
        "#c7384e",   # deep red
        "#9b3049",   # deeper red
        "#862976",   # purple
        "#431533",   # deep purple
        "#030001"    # almost black
    ],
    
    "gradient_teal": ["#e7f8ec", "#c3ecd3", "#9ee0c0", "#71c0a0", "#4eac8d", "#30937f", "#1d8078", "#0a5f65", "#004156"],

    "gradient_red": ["#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#ef6548", "#d7301f", "#b30000", "#7f0000"],

    "gradient_purple": ["#fbf2ff", "#f4d6f8", "#edb8f9", "#da92df", "#c770cf", "#9f4bad", "#78308f", "#5b2477", "#431a63"],

    "gradient_hotpink": ["#f3e7f2", "#edc9ef", "#e9a6e9", "#e081dd", "#d554c8", "#c32bab", "#b3198f", "#940d66", "#730943"],

    "diverging_TealOrange": [  # Diverging color scale from teal to orange
        "#004c4b", "#008786", "#6bbbaf", "#b6e0db",
        "#ffffff",  # neutral midpoint
        "#ffd8b8", "#ffb271", "#e57100", "#8a3b00"
    ],

    "diverging_GreenPink": [  # Diverging color scale from green to pink
        "#276419", "#7fbc41", "#b8e186", "#e6f5d0",
        "#ffffff",  # neutral midpoint
        "#fde0ef", "#de77ae", "#c51b7d", "#8e0152"
    ],

    "diverging_TealPurple": [  # Diverging from teal to purple
        "#004c4b", "#008786", "#6bbbaf", "#b6e0db",
        "#ffffff",
        "#eadaf2", "#d798da", "#b05ab8", "#511a6d"
    ], 

    "diverging_GreenPurple": [  # Diverging from green to purple
        "#276419", "#7fbc41", "#b8e186", "#e6f5d0",
        "#ffffff",
        "#eadaf2", "#d798da", "#b05ab8", "#511a6d"
    ],

    "diverging_BlueRed": [  # Classic diverging blue to red
        "#053061", "#2a77b9", "#64afde", "#b5e4fd",
        "#ffffff",
        "#ffcfb5", "#f4866a", "#c83f3e", "#67001f"
    ]
}   

# Convert each list of hex colors into a `LinearSegmentedColormap` object
# This allows smooth interpolation between the defined colors in plots
for item in SoW_cmap.keys():
    SoW_cmap[item] = LinearSegmentedColormap.from_list(item, SoW_cmap[item])

