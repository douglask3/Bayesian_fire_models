import sys
sys.path.append('.')
sys.path.append('plotting/')
sys.path.append('src/')
sys.path.append('make_input/burnt_area/')
from plot_effect_rr import *
from attribution_where import *
from plot_BA_climateology import *
import imageio

if __name__=="__main__":
    dir1 = "outputs/outputs_scratch/ConFLAME_nrt-attribution-base-2/"
    dir2 = "/time_series/_20-frac_points_0.2/"

    regions = ["Amazon", "Pantanal"] #
    regions = ["Pantanal"]
    region_names = ['Amazonia', 'Pantanal'] #
    region_names = ['Pantanal']
    obs_dir = 'data/data/driving_data_base//'
    obs_file = 'burnt_area_data.csv'
    
    plot_attribution_scatter(regions, "attribution_metrics_era5_base",
                             dir1 = dir1, dir2 = dir2,
                             obs_dir = obs_dir, obs_file = obs_file, 
                             plot_FUN = effect_ratio_and_rr_over_range)
    
    '''plot_attribution_scatter(regions, "attribution_metrics_era5_cf_mean_base",
                             dir1 = dir1, dir2 = dir2, 
                             counterfactual_name = 'counterfactual-metmean',
                             obs_dir = obs_dir, obs_file = obs_file, 
                             plot_FUN = effect_ratio_and_rr_over_range) 
    '''
    outs_era5 = plot_attribution_scatter(regions, "attribution_scatter_era5_base",
                             dir1 = dir1, dir2 = dir2,
                             obs_dir = obs_dir, obs_file = obs_file) 
    
    '''outs_era52 = plot_attribution_scatter(regions, "attribution_scatter_era5_cf_mean_base",
                             dir1 = dir1, dir2 = dir2, 
                             counterfactual_name = 'counterfactual-metmean',
                             obs_dir = obs_dir, obs_file = obs_file) 
    '''
    
    regions = ["Amazon_base", "Pantanal_base"]
    dir2s = ["/samples/_19-frac_points_0.2/", "/samples/_20-frac_points_0.2/"]
    for dir2, region in zip(dir2s, regions):
        from  pathlib import Path
        
        eg_file  = list(Path(dir1 + get_region_info(region)[region]['dir'] + dir2 + '/').rglob('*.nc'))[0]
   
        
        fig, axes = set_up_sow_plot_windows(4, 4, iris.load_cube(eg_file)[0], size_scale = 3)
        for mnth, ax in zip(range(12), axes[0:12]):
            month_idx = '0' + str(mnth + 1) if mnth <9 else str(mnth + 1)
            print(month_idx)
            map_attribution_for_region(dir1, dir2, region, 
                                       temp_filename = '-base-' + month_idx + '-', 
                                       ax = ax, add_cbar = False, month_idx = [month_idx])
        
        img = map_attribution_for_region(dir1, dir2, region, 
                                         temp_filename = '-base-', 
                                         ax = axes[12], add_cbar = False)    
        axes[13].set_visible(False)
        axes[14].set_visible(False)
        axes[15].set_visible(False)
        pos13 = axes[13].get_position()
        pos15 = axes[15].get_position()
        x0 = pos13.x0
        x1 = pos15.x1
        y1 = pos13.y0  # top of the top row
        height = 0.8  # thickness of the colorbar

        cbar_ax = fig.add_axes([x0, y1 + 0.01, x1 - x0, height])  # [left, bottom, width, height]i
        cbar_ax.set_visible(False)

        add_attribubtion_map_cbar(img, cbar_ax)
        
        plt.savefig("figs/attrbution_where-base" + region + ".png", dpi = 300) 
        
        filename = "data/data/driving_data_base/" + region[:-5] +"/burnt_area.nc"
        anomaly, climatology = open_netcdf_and_find_clim(filename)
        BA_anomaly = anomaly[4:16]
        frames = [[], []]
        # Temporary folder for images
        outdir1 = "temp/base_attribution" + region + '/'
        outdir2 = "temp/base_burned_area" + region + '/'
        os.makedirs(outdir1, exist_ok=True)
        os.makedirs(outdir2, exist_ok=True)
        
        for mnth in range(13):
            
            month_idx = f"{mnth+1:02d}"  # 01, 02, ...
            if month_idx == "13":
                month_in = None
            else:
                month_in = [month_idx]

            fig, ax = set_up_sow_plot_windows(1, 1, iris.load_cube(eg_file)[0], size_scale = 6)
            map_attribution_for_region(
                dir1, dir2, region,
                temp_filename = '-base-' + month_idx + '-',
                ax = ax,
                add_cbar = False,   # you might want a colourbar per frame
                month_idx = month_in
            )
            add_attribubtion_map_cbar(img, ax)
            # save the frame
            fname0 = os.path.join(outdir1, f"frame_{month_idx}.png")
            plt.savefig(fname0, dpi=150, bbox_inches="tight")
            plt.close(fig)
            frames[0].append(fname0)        
    
            fig, ax = set_up_sow_plot_windows(1, 1, iris.load_cube(eg_file)[0], size_scale = 6)
            if mnth == 12:
                pcube = BA_anomaly.collapsed('time', iris.analysis.SUM)
                title = month_range(['01', '12'])
            else:
                pcube = BA_anomaly[mnth]
                title = month_range([month_idx])
            
            plot_map_sow(pcube, title,
                        cmap=SoW_cmap['diverging_BlueRed'], 
                        levels = [-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20],
                        extend = 'both', cbar_label = "Burned Area Anomaly (%)",
                        add_cbar = True,
                        ax = ax)

            fname1 = os.path.join(outdir2, f"frame_{month_idx}.png")
            plt.savefig(fname1, dpi=150, bbox_inches="tight")
            plt.close(fig)
            frames[1].append(fname1)
        for i in range(11):
            frames[0].append(fname0)
            frames[1].append(fname1)
        # Stitch into a GIF
        images = [imageio.imread(f) for f in frames[0]]
        imageio.mimsave("figs/" + "attribtion_maps" + region + "months.gif", images, fps=0.75, loop = 0)  # duration = seconds per frame 
        images = [imageio.imread(f) for f in frames[1]]
        imageio.mimsave("figs/" + "Burned_area_maps" + region + "months.gif", images, fps=0.75, loop = 0)  # duration = seconds per frame
