## model info
regions:: ['Pantanal']
model_title::  'attribution-base-localBA-data-SuperSlimmed-17/<<region>>'

## input data paths and filenames

dir_training:: "data/data/driving_data_base/<<region>>/nrt/era5_monthly/"
#dir_training:: "C:/BASE/Data/<<region>>/monthly_era5/"


#y_filen:: "burnt_area.nc"
#y_filen:: "Fire_fraction_Amazonia.nc"
y_filen:: "Fire_fraction_Pantanal.nc"


CA_filen:: None

#Pantanal
x_filen_list:: ["precip", "max_consec_dry", "tas_max", "hurs_min", "Agriculture_fraction_Pantanal", "Pasture_fraction_Pantanal", "Forest_fraction_Pantanal", "Grassland_fraction_Pantanal", "Savanna_fraction_Pantanal", "TV_CFUEL_CORRECTED_MAP", "rural_population_regridded_to_era5", "urban_area_regridded_to_era5",  "Wetland_fraction_Pantanal", "monthly_cg_strokes", "change_Wetland_fraction_Pantanal.nc", "change_Forest_fraction_Pantanal.nc"] #"roadDensity",
#dry_days, hurs_mean and wind
Y_scale:: 0.01

## Model info
model_class:: ConFire
priors:: {'pname': "link-qSpread_mu",'np': 1, 'dist': 'Normal', 'mu': 1.0, 'sigma': 0.5}
priors:: {'pname': "link-qSpread_sigma",'np': 1, 'dist': 'HalfNormal', 'sigma': 0.5}

control_names:: ['Fuel','Moisture', 'Suppression', 'Ignitions']
priors:: {'pname': "control_Direction", 'value': [1, 1, 1, 1]}
priors:: {'pname': "controlID", 'value': [[7, 8, 12], [1, 2, 6, 8, 14], [4], [4, 5, 10, 12]]}
priors:: {'pname': "driver_Direction", 'value': [[1, 1, 1], [-1, -1, -1, -1, 1], [-1], [1, 1, 1, 1]]}
priors:: {'pname': "x0",'np': 4, 'dist': 'Normal', 'mu': 0.0, 'sigma': 10.0}

#betas
priors:: {'pname': "betas",'np': 3, 'dist': 'LogNormal', 'mu': 0.0, 'sigma': 2.0}
priors:: {'pname': "betas",'np': 5, 'dist': 'LogNormal', 'mu': 0.0, 'sigma': 2.0}
priors:: {'pname': "betas",'np': 1, 'dist': 'LogNormal', 'mu': 0.0, 'sigma': 2.0}
priors:: {'pname': "betas",'np': 4, 'dist': 'LogNormal', 'mu': 0.0, 'sigma': 2.0}
#powers
priors:: {'pname': "powers",'np': 3, 'dist': 'LogNormal', 'mu': 1.0, 'sigma': 2.0}
priors:: {'pname': "powers",'np': 5, 'dist': 'LogNormal', 'mu': 1.0, 'sigma': 2.0}
priors:: {'pname': "powers",'np': 1, 'dist': 'LogNormal', 'mu': 1.0, 'sigma': 2.0}
priors:: {'pname': "powers",'np': 4, 'dist': 'LogNormal', 'mu': 1.0, 'sigma': 2.0}


### optimization info
inference_step_type:: "metropolis"
link_func_class:: MaxEnt
niterations:: 1000
cores:: 10

fraction_data_for_sample:: 0.1
min_data_points_for_sample:: 1000

#for testing purposes
#fraction_data_for_sample:: 0.000002
#min_data_points_for_sample:: 100

subset_function:: sub_year_months
#Amazonia - June to October 
#Pantanal - May to Sep
subset_function:: sub_year_range
subset_function_args:: {'months_of_year': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}

#training period
subset_function_args:: {'year_range': [2010, 2026]}
#region_months:: {'Pantanal': [5, 6, 7, 8], 'Amazon': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
region_months:: {'Pantanal': [4,5, 6, 7, 8], 'Amazon': [5, 6, 7, 8, 9]}
grab_old_trace::  True # set to True till you get the code running. 
                      # Then set to False when you start adding in new response curves
### output info
dir_outputs:: 'outputs/outputs_scratch/'

## evaluation info
y_filen_eval:: 'Fire_fraction_Pantanal.nc'
subset_function_eval:: [sub_year_range, sub_year_months]
subset_function_args_eval:: [{'year_range': [2024, 2026]}, {'months_of_year': [4,5, 6, 7, 8]}]
plot_drivers :: True
sample_for_plot:: 50
#if it doenst work delte the sample folder and generate again

plot_control_maps:: True
control_colours:: ['gradient_red', 'gradient_purple', 'gradient_greys', 'gradient_teal']
levels:: [0, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0] 
dlevels:: [-20, -10, -5, -2, -1, -0.1, 0.1, 1, 2, 5, 10, 20]
cmap:: 'gradient_red' 
dcmap:: 'diverging_TealOrange'
#dir_projecting:: "data/data/driving_data_base/<<region>>/nrt/era5_monthly/"
dir_projecting:: "C:/BASE/Data/<<region>>/monthly_era5/"
parallelize:: False

## experiment info
limitation_types:: ['standard']
controls_to_plot:: [0, 1, 2, 3]
#experiment_dir:: ["data/data/driving_data_base/<<region>>/nrt/era5_monthly/", "data/data/driving_data_base/<<region>>/nrt/era5_monthly/CF_mean/||data/data/driving_data_base/<<region>>/nrt/era5_monthly/","data/data/driving_data_base/<<region>>/nrt/era5_monthly/CF/||data/data/driving_data_base/<<region>>/nrt/era5_monthly/"]
#experiment_type:: ['single', 'single', 'ensemble-single']
#experiment_names:: ["factual", "counterfactual-metmean", "counterfactual"]

experiment_dir:: ["data/data/driving_data_base/<<region>>/nrt/era5_monthly/", "data/data/driving_data_base/<<region>>/nrt/era5_monthly/CF/||data/data/driving_data_base/<<region>>/nrt/era5_monthly/"]
experiment_type:: ['single', 'ensemble-single']
experiment_names:: ["factual", "counterfactual"]

time_series_percentiles:: [0.0, 95.0]


#python run_ConFLAME.py namelists/nrt-attribution-base.txt
