# Bayesian-based fire models

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Obtaining Driving Data](#obtaining-driving-data)
- [Configuration Settings](#configuration-settings)
- [Model options](#Model-options)
- [Link distribution options](#Link-distribution-options)
- [Running the Models](#running-the-models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [State of Wildfires report](#State-of-Wildfires-report)

## Introduction
This repository contains a series of Bayesian-based fire models. These models and probablity distribution link functions follow a common inference and sampling workflow to predict fire behaviour based on various driving data. 

### Models 
At the moment, we have two models that target burned area:
*	[ConFire](https://github.com/douglask3/Bayesian_fire_models/blob/main/README/ConFire.md), has been tested and was working as expected at version [v0.1](https://github.com/douglask3/Bayesian_fire_models/releases/tag/SoW23_v0.1), though will need testing again now.
*	[FLAME](https://github.com/douglask3/Bayesian_fire_models/blob/main/README/FLAME.md), is also operational but is located in a separate repository. It needs to undergo testing in our current environment. You can find it at https://github.com/malu-barbosa/FLAME.
*	[ConFLAME](https://github.com/douglask3/Bayesian_fire_models/blob/main/README/ConFLAME.md), tested and working.
  
Additionally, we are in the process of working on INFERNO, which is derived from JULES-ES-INFERNO ([Mangeon et al. 2016]( https://gmd.copernicus.org/articles/9/2685/2016/)) and working on a  fireMIP ensemble optimisation scheme. We are also extending these two models to cover other aspects of fire regimes beyond burned area.

### Link functions

*	[Zero Inflated logit function](https://github.com/douglask3/Bayesian_fire_models/blob/main/README/Zero%20Inflated%20logit%20function), has been tested and is working as expected.
*	[MaxEnt](https://github.com/douglask3/Bayesian_fire_models/blob/main/README/MaxEnt), is also operational but is located in a separate repository. It needs to undergo testing in our current environment. You can find it at https://github.com/malu-barbosa/FLAME.

## Features
- Bayesian inference for fire modelling
- Flexible configuration settings
- Different fire models for relating drivers to burnt area
- Scalable to different data sets and scenarios
- Detailed logging and result output

## Installation
Follow these steps to install the necessary software and dependencies:

1. Clone the repository:
    ```bash
    git clone https://github.com/douglask3/Bayesian_fire_models.git
    ```
    or download the latest version from Zenodo: xxxxxxx
3. Navigate into the project directory:
    ```bash
    cd Bayesian_fire_models
    ```
4. Install the required packages:
    
    ```bash
    conda env create -f environment.yml
    conda activate bayesian-fire-models
    ```

## Obtaining Driving Data
The driving data required for the models are listed [here](https://github.com/douglask3/Bayesian_fire_models/blob/main/README/Datasets). Once you have downloaded the dataset you want, save it to the `data/data/driving_data/` directory within the project folder or update the path in the corresponding configuration file (see below) to its location.

## Configuration Settings
Configuration settings can be found in the text files in `namelists\` dir. [namelists/nrt-evaluation.txt](https://github.com/douglask3/Bayesian_fire_models/blob/main/namelists/Pantanal_example.txt) is quite a good simple example that works for [ConFLAME](https://github.com/douglask3/Bayesian_fire_models/blob/main/README/ConFLAME.md). There are some model speific parameters, so you'll have to check specific under [Model Setup](#model-setup). But this is common to all models.

Each parameter is set using `parameter_name:: parameter_value'. "parameter_name" is a parameter or setting used by the framework. "parameter_value" covers most common Python objects (ints, floats, string, lists, common function or class objects) and some extra objects from sync or this repo. Some fields allow wildcard entries that are set as a list within another parameter and looped over, thereby saving lines in the configuration file. These are indicated in the example, but wildcards are never compulsory. 



| Parameter               | Description                                                   | Compulsory | Example                                                                                      |
|-------------------------|---------------------------------------------------------------|------------|-----------------------------------------------------------------------------------------|
| `regions`               | List of regions to model - useful if running the exact same run over multiple  driving datasets  | No - but required if using `<<region>>` wildcards.  | `['Greece', 'Canada', 'NW_Amazon']`            |
| `model_title`           | Title of the model. Can use `regions` to set up a new run for each region | Yes        | `'ConFire_<<region>>-nrt-evaluation1'`                                  |
| `dir_training`          | Directory containing driving/target data for training         | Yes        | `"data/data/driving_data/<<region>>/nrt/period_2013_2023/"`                                  |
| `y_filen`               | Netcdf filename for target variable, either within  `dir_training` or relative for pull path is starting with '.' or '~'  | Yes        | `"burnt_area.nc"`           |
| `CA_filen` | Netcdf filename for area weights for model training - i.e. if you want to weigh some grid cells more than others.                     | No         | `None`                      |
| `x_filen_list`          | List of filenames for predictor variables                     | Yes        | `["VOD-12monthMax.nc", "VOD-12Annual.nc", "Fuel-Moisture-Live.nc", ...]`                     |
| `model_class`           | The name of the fire model we are going to use. Coded up in `fire_model/'.  See  [Model options](#Model-options)    | Yes        | `ConFire`                                                              |
| `link_func_class`       | The name of the link function/assumed error distribution. Coded up in `link_distribution/`. See [Link distribution options](#Link-distribution-options)  | Yes        | `zero_inflated_logit`                              |
| `priors`                | Priors for Bayesian inference. These use their own funky definition rather than a standard Python object but are defined using a dict, with the following fields that need filling out, depending on prior type: | At least one. This is model and link function specific. These are parameters used in either the model or link function | Some examples below: |
|     |  `pname` is the parameter name used by the model. See the specific model for details. if `npame` starts with the string `link-`, then it is a parameter used in the link function. Again, see the specific link function for info. |   | `{'pname': "link-sigma", 'np': 1, 'dist': 'HalfNormal', 'sigma': 0.5}`  |
|     |  `dist`. The prior can be based on a pymc prior distribution. If it is, you can use any in the pymc library: [https://www.pymc.io/projects/docs/en/stable/api/distributions.html](https://www.pymc.io/projects/docs/en/stable/api/distributions.html). To use one of these, enter a pymc distribution name in the entry `dist`. You then enter each input used by that distribution as a new dict item (e.g. `upper` and `lower` in the example). `np` is normally an optional field (though some models or link functions may require it) that says how many times you need this distribution. See model or link function for specifcs | | `{'pname': "Fmax", 'np': 1, 'dist': 'Uniform', 'lower': 0.00001, 'upper': 0.9999}`  |
|                         |                                                         |            | `{'pname': "x0", 'np': 6, 'dist': 'Normal', 'mu': 0.0, 'sigma': 10.0}`                       |
|                         |  Sometime you want more certain priors (i.e, these are fixed and do not change  during optimization). For this, the only addition dict field you need is `value` and the value matches the python object required by that parameter in the model you are using    |            | `{'pname': "driver_Direction", 'value': [[1, 1], [1, 1], [1, 1, -1], [1, 1, 1], [1, 1, 1, 1], [1]]}`          |
 `niterations` | This is the number of iterations or samples the model is used when training the data. When sampling the optimised distribution, you won't be able to sample more than niterations x cores, so make sure the two parameters are set high enough                                          | Yes        | `1000`                                                                                         |
| `inference_step_type`   | This is the type of step used for pymc sampler. If not provided, default is `NUTS`  | No        | `NUTS`. options can be found [here](https://github.com/douglask3/Bayesian_fire_models/blob/main/README/pymc_step_options) |
| `cores`                 | This is the number of chains (optimisation attempts) AND the number of computer cores on platforms where multicore-ing in a thing   | Yes        | `100`                                                                   |
| `fraction_data_for_sample` | The fraction of the target data used for optimization | Yes        | `0.5`                                                                                        |
| `subset_function`       | Function to subset the data. See [Subsetting data function](https://github.com/douglask3/Bayesian_fire_models/blob/main/README/Subsetting_data_function.md) for options      | No, but you can have multiple too        | `sub_year_months`                                                                            |
|                         |                                                               |            | `sub_year_range`                                                                             |
| `subset_function_args`  | Arguments for subset function. As a sict with inputs for the subset function. One line for each function above in order.   | If  any of the the subset functions requires them        | `{'months_of_year': [4, 5, 6, 7, 8]}`                                                        |
|                         |                                                               |            | `{'year_range': [2013, 2017]}`                                                               |
| `subset_function_args_eval` | This can replace the subset function argument when you are sampling the posterior. Useful for out-of-sample experiments.  | No        | `{'months_of_year': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}`                                 |
|                         |                                                               |            | `{'year_range': [2018, 2023]}`                                                               |
| `grab_old_trace`        | Whether to use old traces and samples from previous optimisation attempts. If `True`, then it will attempt to find old files that have a similar look to this configuration file. But it's not foolproof so if in doubt, set to `False`                   | No         | `True`                                                                                       |
| `dir_outputs`           | Directory for outputs                                         | Yes        | `'outputs/'`                                                                                 |
| `sample_for_plot`       | Number of samples from the optimised trace for evaluation and plotting                                | Yes        | `100`                                                                                        |
| `levels`                | colourmap levels for burnt area maps                                      | Yes        | `[0, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]`                                                        |
| `dlevels`               | colourmap levels for difference in burnt area maps                                 | Yes        | `[-20, -10, -5, -2, -1, -0.1, 0.1, 1, 2, 5, 10, 20]`                                        |
| `cmap`                  | Colormap for plots                                            | Yes        | `'OrRd'`  |
| `dcmap`                 | Differential colormap for plots                               | Yes        | `'RdBu_r'` |
| `limitation_types`      | The type of limitations that will be outputed                 | No         | current options are: `'standard'`, `'potential'`, `'potential_climateology'` |
| `controls_to_plot`      | which control ids to analyes and plot                               | No        | `[0, 1]` |
| `time_series_percentiles`| The percentile of BA of which to produce time series ocer (i.e 95.0 analysised gridcells with top 5% BA only) | No        | `[0.0, 95.0]` |


## Model options
ConFire and ConFLAME are working. FLAME works in the original repo: [https://github.com/malu-barbosa/FLAME](https://github.com/malu-barbosa/FLAME) but is untested here. Others are under development

Available models:
* [ConFLAME](https://github.com/douglask3/Bayesian_fire_models/blob/main/README/ConFLAME.md)
* [ConFire](https://github.com/douglask3/Bayesian_fire_models/blob/main/README/ConFire.md)
* [FLAME](https://github.com/douglask3/Bayesian_fire_models/blob/main/README/FLAME.md)

## Link distribution options 
ConFire is working. FLAME works in the original repo: [https://github.com/malu-barbosa/FLAME](https://github.com/malu-barbosa/FLAME) but is untested here. Others are under develop,ent
There are a couple of link distributions.  Zero Inflated logit function works just fine, MaxEnt needs some updates
*	[Zero inflated logit function](https://github.com/douglask3/Bayesian_fire_models/blob/main/README/Zero_inflated_logit_function.md)
*	[MaxEnt](https://github.com/douglask3/Bayesian_fire_models/blob/main/README/MaxEnt)
*	`normal` technical runs but is having issues.

## Running the Models
Over the next few months, we will generalise the model's running so it works the same way for any more setups, and you define the name list at runtime. For now, though, we have basic model execution files for ConFire: one for near real-time studies (`run_ConFire-NRT.py`) and one for ISI-MIP-driven runs for attribution and future projections (`run_ConFire.py`). For each, open the file, under `__main__`, make sure the variable is set to your namelist, and then run:

* for isimip
```bash
python run_ConFire.py
```

* for nrt
```bash
python run_ConFire-NRT.py
```

## Results
The results will be saved in the `outputs/` directory, though this can be updated in the namelists. There are some model specific files that will get produced, but common across all models are:

## Contributing
Contributions are welcome! Please read the [CONTRIBUTING.md](README/CONTRIBUTING.md) file for guidelines on how to contribute.

## License
This project is licensed under the  GNU GENERAL PUBLIC LICENSE version 3 License. See the [LICENSE](LICENSE) file for details.

## Contact
Please contact [Dougas Kelley] at [doukel@ceh.ac.uk] for any questions or issues.

## State of Wildfires report
The State of Wildfires report has been a major driver of this development. Here's some info for anyone whose found their way here and wants to perform those runs again

### 2023/24
To reproduce results in the State of Wildfire's 2023/24 report, make sure you have the version at tag SoW23_v0.1 (0.1 because the paper is still in review). Or download the Zenodo archived code that can be found with the paper: https://doi.org/10.5194/essd-2024-218, along with dataset information. The achived driving and output data is [https://doi.org/10.5281/zenodo.11420743](https://doi.org/10.5281/zenodo.11420743).

The results from this paper were obtained using the [Running the Models](#running-the-models) commands above. It combines [ConFire](https://github.com/douglask3/Bayesian_fire_models/blob/main/README/ConFire.md) with the [zero inflated logit function](https://github.com/douglask3/Bayesian_fire_models/blob/main/README/Zero_inflated_logit_function.md) logistic function The namelists are already set within these files, but you may need to update paths in the following in the `namelists` directory:
* `isimip.txt` - used for attribution and future projections
* `isimip-evaluation.txt` used to evaluate the configuration for attribution and future projections
* `nrt.txt` used for 2023 fire season driver analysis
* `nrt-evalution.txt` - used to evaluate the driver analysis configuration


