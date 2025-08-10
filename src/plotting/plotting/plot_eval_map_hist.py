import sys
sys.path.append('SoW_info/')
from state_of_wildfires_colours  import SoW_cmap
from state_of_wildfires_region_info  import get_region_info

from plot_maps import *
from  constrain_cubes_standard import *
from bilinear_interpolate_cube import *



plot_map_sow(cube, "", 
                    cmap=SoW_cmap['diverging_TealOrange'], 
                    levels=levels_BA_obs,#region_info['Anomoly_levels'], 
                    ax=axes[0], cbar_label = "Burned Area Anomaly (%)")
