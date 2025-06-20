import xarray as xr
from pathlib import Path
import re
from collections import defaultdict

def run_for_region(region):

    # Input and output directories
    base_dir = Path('data/data/driving_data2425/' + region + '/nrt/era5_monthly/CF/')
    output_dir = Path('data/data/driving_data2425/' + region + '/nrt/era5_monthly/CF_mean/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Gather all .nc files recursively
    nc_files = list(base_dir.rglob('*.nc'))
    
    # Regex to separate variable and number. Adjust this if your filename pattern differs.
    # Assumes var is alphabetic and number is numeric at end of basename before ".nc"
    var_regex = re.compile(r'([A-Za-z_]+)(\d+)\.nc$')
    
    # Group files by var
    files_by_var = defaultdict(list)
    for f in nc_files:
        m = var_regex.search(f.name)
        if m:
            var = m.group(1)
            files_by_var[var].append(f)
    
    # Process each var group
    for var, files in files_by_var.items():
        print(f'Processing variable: {var}, {len(files)} files')
        # Open them as one combined Dataset
        ds_list = [xr.open_dataset(f) for f in files]
        # Average across all datasets
        mean_ds = sum(ds_list) / len(ds_list)
        
        # Save to output
        output_file = output_dir / f"{var}_mean.nc"
        mean_ds.to_netcdf(output_file)
        print(f"Saved: {output_file}")

        # Close all the source datasets
        for ds in ds_list:
            ds.close()

regions = ['Amazon', 'Congo', 'LA', 'Pantanal']
for region in regions: 
    run_for_region(region)
