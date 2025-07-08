import xarray as xr
from pdb import set_trace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import fnmatch
import pickle
from scipy.stats import genpareto

import sys
sys.path.append('.')
sys.path.append('src/')
sys.path.append('SoW_info/')
from state_of_wildfires_colours  import SoW_cmap
from state_of_wildfires_region_info  import get_region_info

def extract_years(df, years, mnths, ext = "-01T00:00:00"):
    """
    Extracts and averages values from a DataFrame across specified months and years.

    Parameters:
    ----------
    df : pandas.DataFrame
        A DataFrame with columns labeled by timestamp strings (e.g., '2023-01-01T00:00:00') 
        and rows representing ensemble members or samples.
    years : list of str or int, or None
        The years to include (e.g., [2023, 2024]). If None, all years found in column names are used.
    mnths : list of str
        The months to average over (e.g., ['01', '02', '03'] for January–March).
    ext : str, optional
        A string pattern representing the suffix to match in column names (default is '-01T00:00:00'), 
        though it's overridden in favor of a wildcard match.

    Returns:
    -------
    np.ndarray
        A 1D numpy array with the average value for each row (e.g., ensemble member), 
        computed across the selected months for each year. The result is flattened to combine years.
    
    Notes:
    -----
    - Column matching uses wildcards via `fnmatch` to allow flexibility in timestamp formats.
    - Designed for use in ensemble climate/fire datasets where time is encoded in column headers.
    - Useful for computing seasonal means (e.g., JFM) per year, per ensemble member.
    """
    if years is None:
        years = np.unique([col[0:4] for col in df.columns[1:]])
    # Reshape: group columns by year
    avg_per_year = []
    for year in years:
        cols_this_year = [
            col for col in df.columns
            for month in mnths
            if fnmatch.fnmatch(col, f"{year}-{month}*")
        ]
    
        avg_per_year.append(df[cols_this_year].mean(axis=1))
    
    return np.array(avg_per_year).flatten()

def flatten(xss):
    """
    Flattens a nested list (list of lists) into a single list.

    Parameters:
    ----------
    xss : list of lists
        A list where each element is itself a list.

    Returns:
    -------
    list
        A single flattened list containing all elements from the sublists, in order.
    """
    return [x for xs in xss for x in xs]


def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

def scale2upper1(y):
    return 1-np.exp(-y * (-np.log(0.5)))

def scale2upper1_inverse(z):
    return -np.log(1 - z) / np.log(2)

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
    if ytick_labels is None:
        if ylim is None or ylim[0] < 0.2:
            ylim = [0,1]
            ytick_labels = np.array([0, 0.2, 0.5, 1, 2, 5])
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
    ytick_labels_txt = [f"{v:.2f}" for v in ytick_labels] + ['']
    if len(np.unique(ytick_labels_txt)) < len(ytick_labels):
        ytick_labels_txt = scale2upper1_labels(ytick_labels) + ['']
    
    # Step 3: Apply to plot
    try:
        ax.set_yticks(yticks_transformed)
        ax.set_yticklabels(ytick_labels_txt)
    except:
        set_trace()
    ax.set_ylim(ylim)

def plot_kde(x, y, xlab, ylab, cmap_name = "gradient_hues_extended", ax = None, *args, **kw): 
    """
    Creates a filled 2D kernel density estimate (KDE) plot for two input variables.

    Parameters:
    ----------
    x : array-like
        Values for the x-axis.
    y : array-like
        Values for the y-axis.
    xlab : str
        Label for the x-axis and corresponding DataFrame column.
    ylab : str
        Label for the y-axis and corresponding DataFrame column.
    cmap_name : str, optional
        Name of the colormap to use from the SoW_cmap dictionary. Default is "gradient_hues_extended".
    *args, **kw : additional arguments
        Additional arguments passed to `sns.kdeplot`.

    Notes:
    -----
    - Assumes a dictionary `SoW_cmap` is available in the global scope, with named colormaps.
    - Produces a filled density plot using seaborn’s `kdeplot`.

    Example:
    -------
    #>>> plot_kde(x_values, y_values, "Factual", "Counterfactual", cmap_name="SoW_gradient")
    """
    if ax is None: ax  = plt.gca()
    y = scale2upper1(y)
    df = pd.DataFrame({xlab: x, ylab: y})
    
    sns.kdeplot(data=df, x=xlab, y=ylab, fill=True, 
                cmap=SoW_cmap[cmap_name], ax = ax, *args, **kw)
    scale2upper1_axis(ax)

def plot_fact_vs_counter(factual_flat, counterfactual_flat, obs, ax = False): 

    """
    Plots a 2D KDE of Factual vs. Counterfactual burned area values, along with a 1:1 reference line 
    and an observed value marker.

    Parameters:
    ----------
    factual_flat : array-like
        Flattened array of burned area values under factual conditions.
    counterfactual_flat : array-like
        Flattened array of burned area values under counterfactual
        (no-climate-change) conditions.
    obs : float
        Observed burned area value to highlight on the plot.
    ax : matplotlib.axes.Axes, optional
        Optional axis object to plot on. If False, uses the default current axis.

    Notes:
    -----
    - Uses a highly skewed KDE level distribution to emphasize differences at low burned area.
    - Applies logarithmic KDE scaling to better distinguish dense low-value regions.
    - Adds a 1:1 line to visualize agreement between factual and counterfactual conditions.
    - Draws vertical and horizontal red dashed lines at the observed value.
    """

    x = np.linspace(0, 1, 20)
    log_levels = x**(8)  # try 3, 5, 7 for increasingly strong bias
    plot_kde(factual_flat, counterfactual_flat, "factual", "counterfactual",
             levels=log_levels, log_scale = True, thresh=1e-4, ax = ax)

    plt.plot([0.0000000001, 100], [0.0000000001, 100], 'k--', label='1:1 Line')
    plt.ylabel("Counterfactual Burned Area")
    plt.xlabel("Factual Burned Area")
    plt.title("Factual vs Counterfactual Burned Area")
    
    plt.axvline(obs, color='red', linestyle='--', label='Observed Burned Area')
    
    plt.grid(True)

def fit_gpd_tail(data, threshold_quantile=0.90):
    """Fits GPD to upper tail above a given quantile threshold."""
    data = data[~np.isnan(data)]
    threshold = np.quantile(data, threshold_quantile)
    tail_data = data[data > threshold] - threshold
    if len(tail_data) < 5:
        return None  # too few values to fit
    params = genpareto.fit(tail_data)
    return threshold, params

def estimate_tail_prob_gpd(obs, threshold, params):
    """Estimates P(x > obs) from GPD tail."""
    if obs <= threshold:
        return 1.0  # obs not in tail
    excess = obs - threshold
    return genpareto.sf(excess, *params)

def plot_fact_vs_ratio(factual_flat, counterfactual_flat, obs, plot_name, ax = None):
    """
    Plots a 2D KDE of Factual Burned Area vs. Relative Effect Ratio, quantifying the 
    percentage change in burned area due to climate change or other factors.

    The effect ratio is defined as:
        100 * (factual - counterfactual) / max(factual, counterfactual),
    and is clipped between -100 and 100, where 0 indicates no change.

    Parameters:
    ----------
    factual_flat : array-like
        Flattened array of burned area values under factual (actual) conditions.
    counterfactual_flat : array-like
        Flattened array of burned area values under counterfactual (e.g., no-climate-change) conditions.
    obs : float
        Observed burned area value used to define thresholds and draw reference lines.
    plot_name : str
        Label for the y-axis of the plot, indicating what effect is being shown.
    ax : matplotlib.axes.Axes, optional
        Axis object to plot on. If None, the current axis is used.

    Returns:
    -------
    numpy.ndarray
        Array of effect ratio values for cases where factual > observed, used for further statistical analysis.

    Notes:
    -----
    - Uses a KDE with logarithmic contour levels to visualize the joint distribution.
    - Draws vertical and horizontal lines to mark the observed value and neutral (zero) effect.
    - Annotates the plot with key percentiles of the estimated climate change effect, 
      p-value for positive change, and the risk ratio (how much more frequently extreme fire occurs).
    - Applies clipping to avoid spurious ratios in very small values.
    - Replots with clipping on high observed values for visual emphasis.

    Example:
    --------
    #>>> plot_fact_vs_ratio(factual, counterfactual, obs_value, "Relative Change", ax=ax)
    """
    if obs > factual_flat.max():
        counterfactual_flat = counterfactual_flat * obs * 1.1/factual_flat.max()
        factual_flat = factual_flat * obs * 1.1/factual_flat.max()
    
    effect_ratio = factual_flat/counterfactual_flat
    
    
    x = np.linspace(0, 1, 20)
    log_levels = x**(5)  # try 3, 5, 7 for increasingly strong bias
    
    xmin = factual_flat.max()/1000
    test = factual_flat > xmin
    
    plot_kde(factual_flat[test], effect_ratio[test], "factual", "effect ratio",bw_adjust = 1,
             levels=log_levels, thresh=1e-4, clip=((0.0,factual_flat.max()), (0, 1)),
             ax = ax) 
    
    plot_kde(factual_flat[test], effect_ratio[test], "factual", "effect ratio",bw_adjust = 1,
             levels=log_levels, thresh=1e-4, clip=((obs,factual_flat.max()), (0, 1)),
             ax = ax)
    ax.axhline(0.5, color='k', linestyle='--')#, label='No Change (Ratio = 1)')

    ax.axvline(obs, color='red', linestyle='--', label='Observed Burned Area')
    
    mask = factual_flat > obs
    if (np.sum(mask) < 10):
        mask = factual_flat > np.sort(factual_flat)[-10]
    percentile = [5, 50, 95]
    if np.sum(mask) == 0: 
        return
    
    cc_effect = np.percentile(effect_ratio[mask], percentile)
    cc_effect = np.round(cc_effect, 2)
    #cc_effect = np.round(1.0+cc_effect/(1.0-cc_effect), 2)
    
    pv = np.mean(effect_ratio[mask]>1) + 0.5/np.sum(mask)
    pv = np.round(pv, 2)
    if (pv > 0.99):
        pv = str(">99%")
    else:
        pv = str(int(pv*100)) + '%'

    # Fit GPD to factual and counterfactual
    f_res = fit_gpd_tail(factual_flat, 0.95)
    c_res = fit_gpd_tail(counterfactual_flat, 0.95)

    # If GPD could be fitted
    if f_res and c_res:
        f_thresh, f_params = f_res
        c_thresh, c_params = c_res

        pf = estimate_tail_prob_gpd(obs, f_thresh, f_params)
        pc = estimate_tail_prob_gpd(obs, c_thresh, c_params)

        rr = pf / pc
    else:
    #rr_gpd = np.nan  # or fallback to empirical estimate
        rr = np.sum(factual_flat>obs)/np.sum(counterfactual_flat > obs)
    
    
    rr = np.round(rr, 2)
    def cc_str(cc):
        if cc > 100.0:       
            cc = '> 100'
        else:
            cc = str(cc)
        return(cc)
    percentile_text = [str(pc) + "%:\n" + cc_str(cc) for pc, cc in zip(percentile, cc_effect)]
    
    ax.text(0.3, 0.35, "Climate change impact (likelihood: " + pv + ")", 
              transform=ax.transAxes, axes = ax)
    
    for i in range(len(percentile_text)):
        ax.text(0.3 + i*0.18, 0.2, percentile_text[i], transform=ax.transAxes)
    ax.text(0.3, 0.09, "Risk Ratio:", transform=ax.transAxes)
    ax.text(0.3, 0.02, rr, transform=ax.transAxes)
    
    ax.set_xlabel(" ")
    ax.set_ylabel(plot_name)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    return effect_ratio[mask]



def plot_for_region(region, metric, plot_FUN, 
                    dir1, dir2, obs_dir, obs_file, 
                    factual_name = "factual", counterfactual_name = "counterfactual",
                    all_mod_years = False, add_legend = False,
                    *args, **kw):
    region_info = get_region_info(region)[region]
    
    years = region_info['years']
    mnths = region_info['mnths']
    # Load the data
    dir = dir1 + region + dir2 + '/'
    try:    
        factual = pd.read_csv(dir + factual_name + "-/" + metric + \
                                     "/members/absolute/Evaluate.csv")
        counterfactual = pd.read_csv(dir + counterfactual_name + \
                              "-/" + metric + "/members/absolute/Evaluate.csv")
    except:
        factual = pd.read_csv(dir + factual_name + "-/" + metric + \
                                     "/points-Evaluate.csv")
        counterfactual = pd.read_csv(dir + counterfactual_name + \
                              "-/" + metric + "/points-Evaluate.csv")
    obs = pd.read_csv(obs_dir + '/' + region + '/' + obs_file)
    obs = obs[['time', 'mean_burnt_area', 'p95_burnt_area']]
    
    # Extra years and flatten the arrays to 1D
    if all_mod_years:
        mod_years = None
    else:
        mod_years = years
    factual_flat = extract_years(factual, mod_years, mnths)/len(mnths) + 0.000000001
    counterfactual_flat = extract_years(counterfactual, mod_years, mnths)/len(mnths)\
                                 + 0.000000001
    
    obs = extract_years(obs.set_index('time').T, years, mnths, '-15')
    
    factual_flat0 = factual_flat.copy()
    if metric == 'mean':
        obs = obs[0]#*20#*33.0
        plot_name = region_info['shortname']
    else:
        plot_name = ""
        obs = obs[1]
    '''    if factual_flat.max() < 1:
            factual_flat = factual_flat * 150.0
            counterfactual_flat = counterfactual_flat * 150.0
        plot_name = ""
    if obs > factual_flat.max() and factual_flat.max() < 1:
        factual_flat = factual_flat * 150
        counterfactual_flat = counterfactual_flat * 150
    '''
    
    factual_flat = factual_flat * 100
    counterfactual_flat = counterfactual_flat * 100
    
    out = plot_FUN(factual_flat, counterfactual_flat, obs, plot_name = plot_name, *args, **kw)
    
    if add_legend:
        plt.legend()

    return out

def plot_attribution_scatter(regions, figname, plot_FUN = plot_fact_vs_ratio,
                             *args, **kw):
    """
    Loads and prepares burned area data for a specified region and metric, then
    calls a user-defined plotting function to visualize the comparison between 
    factual and counterfactual scenarios.

    Parameters:
    ----------
    region : str
        Name of the region to plot.
    metric : str
        Metric to visualize, e.g., 'mean' or 'extreme'. Affects observation indexing and unit scaling.
    plot_FUN : callable
        A plotting function to be applied to the flattened factual and counterfactual data,
        typically something like `plot_fact_vs_ratio` or `plot_fact_vs_counter`.
    dir1 : str
        Path prefix to the directory containing model outputs.
    dir2 : str
        Path suffix after the region name to complete the directory path.
    obs_dir : str
        Directory containing observational data.
    obs_file : str
        Name of the observational data file (within region subfolder).
    factual_name : str, default="factual"
        Subdirectory or file prefix for the factual scenario data.
    counterfactual_name : str, default="counterfactual"
        Subdirectory or file prefix for the counterfactual scenario data.
    all_mod_years : bool, default=False
        If True, use all years from the model data instead of region-specific years.
    add_legend : bool, default=False
        If True, adds a legend to the resulting plot.
    *args, **kw :
        Additional arguments passed to the plotting function.

    Returns:
    -------
    Any
        Output of the plotting function, typically a numpy array of calculated effects.

    Notes:
    -----
    - Automatically handles unit adjustments if max values are small (< 1) by scaling to percentages.
    - Adjusts observed values if they are larger than the model values, to ensure visual comparability.
    - Uses metadata (months and years) defined in `get_region_info(region)`.
    - Loads model data and observations using pandas, and flattens multi-year monthly time series.
    - If `metric` is not 'mean', it assumes it's an extreme value and processes the second observation entry.

    Example:
    --------
    #>>> plot_for_region("Amazon", "mean", plot_fact_vs_ratio, 
                         dir1="/data/", dir2="/outputs", 
                         obs_dir="/obs", obs_file="obs.csv")
    """
    metrics = ["mean", 'pc-95.0']
    fig, axes = plt.subplots(len(regions), 2, figsize=(12, len(regions)*3.5))
    out = []
    for i, metric in enumerate(metrics):
        outi = []
        for j, region in enumerate(regions):
            if len(regions) == 1:
                ax = axes[i]
            else:
                ax = axes[j, i]
            print(region)
            outi.append(plot_for_region(region, metric, plot_FUN = plot_FUN, ax = ax, 
                            *args, **kw))
        out.append(outi)
    
    fig.text(0.05, 0.5, "Amplification Factor", ha='right', va='center', fontsize=12, rotation=90)
    fig.text(0.33, 0.9, "Entire region", ha='center', va='bottom', fontsize=14)
    fig.text(0.73, 0.9, "High burnt areas", ha='center', va='bottom', fontsize=14)
    fig.text(0.5, 0.09, "Factual burned area (%)", ha='center', va='top', fontsize=12)
    
    plt.savefig('figs/' + figname + ".png")
    return out

def add_violin_plot(df, df_type, ax, title):
    data = df[df["Impact Type"] == df_type]
    sns.violinplot(
        data=data,
        cut = 0.0,
        x="Region", y="Amplification Factor", hue="Source",
        split=False, inner="quartile", palette=["#f68373", "#c7384e", "#fc6",  "#862976", "#cfe9ff"], 
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel('')
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    
    ax.axhline(0.5, color='k', linestyle='--')
    
    scale2upper1_axis(ax)

    regions = data['Region'].unique()
    
    sources = data['Source'].unique()
    n_sources = len(sources)

    # 2. Calculate position offsets for sub-violins within each Region group
    positions = []
    for i, region in enumerate(regions):
        for j in range(n_sources):
            offset = -0.3 + j * (0.6 / (n_sources - 1))  # Spread violins within 0.8 width
            positions.append(i + offset)

    #tick_positions = range(len(ax.get_xticks()))
    #tick_labels = ax.get_xticklabels()
    arr = data["likelihood"].values
    custom_labels, idx = np.unique(arr, return_index=True)    
    custom_labels = custom_labels[np.argsort(idx)]
    custom_labels = np.round(custom_labels*100)
    
    y_min = 0.0#data["Amplification Factor"].min()
    for x, label in zip(positions, custom_labels):
        if label > 99:
            label_str = '>99%'
        else:
            label_str = str(label)[0:2] + '%'
        ax.text(x-0.05, 0.4, label_str, ha='center', va='top', fontsize=8, rotation=45)
    
    
#    for i, label in enumerate(custom_labels):
#        ax.text(i, y_min, str(label)[0:2] + '%', ha='center', va='top', fontsize=10, color='black')
    

if __name__=="__main__":
    
    dir1 = "outputs/outputs_scratch/ConFLAME_nrt-attribution9/"
    dir2 = "-2425/time_series/_19-frac_points_0.5/"

    regions = ["Amazon", "Pantanal",  "LA", "Congo"]
    region_names = ['Northeast Amazonia', 'Pantanal and Chiquitano', 
                    'Southern California','Congo Basin']
    #retgions = {key: regions_info[key] for key in region_names if key in regions_info}
    obs_dir = 'data/data/driving_data2425//'
    obs_file = 'burnt_area_data.csv'
    
    outs_era5 = plot_attribution_scatter(regions, "attribution_scatter_era5_2425",
                             dir1 = dir1, dir2 = dir2,
                             obs_dir = obs_dir, obs_file = obs_file) 
    
    outs_era52 = plot_attribution_scatter(regions, "attribution_scatter_era5_cf_mean_2425",
                             dir1 = dir1, dir2 = dir2, 
                             counterfactual_name = 'counterfactual-metmean',
                             obs_dir = obs_dir, obs_file = obs_file) 
    
    dir1 = "outputs/outputs_scratch/ConFLAME_isimip_attribution/ConFLAME_"
    dir2 = "-2425/time_series/_15-frac_points_0.5/"
    dir1 = "outputs/outputs_scratch/ConFLAME_nrt-isimip_large/ConFLAME_"
    dir2 = "-2425/time_series/_16-frac_points_0.5/"
    dir1 = "outputs/outputs_scratch/ConFLAME_isimip_attribution-LUC2/"

    outs_isimip = plot_attribution_scatter(regions, 
                             "attribution_scatter_isimip_2425",
                             dir1 = dir1, dir2 = dir2,
                             obs_dir = obs_dir, obs_file = obs_file, all_mod_years = True) 

    outs_human = plot_attribution_scatter(regions, 
                             "attribution_scatter_isimip_human_2425",
                             dir1 = dir1, dir2 = dir2,
                             obs_dir = obs_dir, obs_file = obs_file, 
                             factual_name = "counterfactual", 
                             counterfactual_name = "early_industrial",
                             all_mod_years = True) 

    outs_all = plot_attribution_scatter(regions, 
                             "attribution_scatter_isimip_all_2425",
                             dir1 = dir1, dir2 = dir2,
                             obs_dir = obs_dir, obs_file = obs_file, 
                             factual_name = "factual", 
                             counterfactual_name = "early_industrial",
                             all_mod_years = True) 

    outs_combined = []
    for ii in range(len(outs_era5)):
        outi = []
        for jj in range(len(outs_era5[ii])):
            if jj % 2 == 0:
                out_e = outs_era52
            else:
                out_e = outs_era5
            try:
                outi.append((np.random.choice(out_e[ii][jj], 1000) + \
                np.random.choice(outs_isimip[ii][jj], 1000))/2.0)
            except:
                outi.append(np.random.choice(out_e[ii][jj], 1000))
        outs_combined.append(outi)

    
    f = open('temp/store.pckl', 'wb')
    pickle.dump([outs_era5, outs_era52, outs_isimip, outs_combined, outs_human, outs_all], f)
    f.close()
    
    f = open('temp/store.pckl', 'rb')
    outs_era5, outs_era52, outs_isimip, outs_combined, outs_human, outs_all = pickle.load(f)
    f.close()

    # Example labels for the sources
    sources = ['Anthropogenic climate forcing', 
              # 'Climate (ERA5 HadGEM means)',
               'Total climate forcing', #'Climate (Combined)', 
               'Socio-economic factors',
               'All forcings']

    # Group data
    all_sources = [outs_era5, outs_isimip, outs_human, outs_all]

    # Flatten into long-form dataframe for seaborn
    records = []
    for region_idx, region in enumerate(region_names):
        for source_idx, source_name in enumerate(sources):
            for kind_idx, kind in enumerate(["Mean", "Extreme"]):  # 0 = mean, 1 = extreme
                samples = all_sources[source_idx][kind_idx][region_idx]
                
                if np.max(samples) > 1.0:
                    liki =  np.mean(samples>1.0)
                else:
                    liki =  np.mean(samples>0.5)
                if samples is None: samples = np.array([1.0, 1.0])
                samples = scale2upper1(samples)
                for val in samples:
                    records.append({
                        'Region': region,
                        'Source': source_name,
                        'Impact Type': kind,
                        'Amplification Factor': val,
                        'likelihood': liki
                    })
    
    df = pd.DataFrame.from_records(records)
    
    # Set up the plot
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    add_violin_plot(df, "Mean", axes[0], "Regional burned areas")
    add_violin_plot(df, "Extreme", axes[1], "Sub-regional extremes")
    axes[0].legend_.remove()
    axes[1].legend(loc="lower left", ncol = 2)

    # Tidy up
    plt.tight_layout()
    plt.savefig("figs/attribution-summery.png")
    
