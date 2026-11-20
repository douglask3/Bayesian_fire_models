import sys
sys.path.append('../../libs/')
sys.path.append('libs/')
sys.path.append('make_inputs/nrt/')
sys.path.append('make_inputs/nrt/src/')

from download_era5_cds import *
from HadGEM_extract import *
from make_er5_extra_vars import *
from regrid_hyde import *
from make_factual_counter import *
import pickle
try:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
except:
    pass


def make_nrt_for_region(region_names, data_dir, shapeinfo):
    
    out_dir = data_dir + "/nrt_raw/"
    if not isinstance(region_names, list):
        region_names = [region_names]
    
    run_era5_download_for_report(shapeinfo, region_names,  out_dir)
    make_era5_extra_vars(data_dir + '/nrt_raw/', region_names, shapeinfo = None)
    
    temp_dir = data_dir + "/hadgem_nrt2/"
    
    for start_year in start_years:
        HadGEM_extract_variables(experiments, variables, start_year, dir, sub_dir,
                          out_dir, temp_dir,
                          region_names = region_names,
                          shapeinfo = shapeinfo)
    
        
        make_all_variable_inputs(datadir = out_dir, regions = region_names, 
                                 hadgem_start_year = start_year)
    
    regrid_hyde_for_regions(region_names, data_dir) 


def make_nrt_for_region_wrapper(kwargs):
    try:
        make_nrt_for_region(**kwargs)
        return (kwargs, "success")
    except Exception as e:
        print(f"[ERROR] Experiment failed with kwargs={kwargs['name']}:\n{e}")
        import traceback
        traceback.print_exc()
        return (kwargs, f"error: {e}")

def make_nrt_inputs(data_dir, shapeinfo, regions):
    if isinstance(regions, list):
        args_list = [dict(region=region,
                          data_dir = data_dir, 
                          shapeinfo = shapeinfo) for region in regions]

        try:
            with get_context("spawn").Pool(processes=4) as pool:
                pool.map(make_nrt_for_region_wrapper, args_list)
        except:
            for args in args_list:
                make_nrt_for_region_wrapper(args)
    else:
        make_nrt_for_region(regions, data_dir, shapeinfo)    

if __name__=="__main__":  
    data_dir = "data/data/driving_data2627/"


    shapeinfo = [[contrain_coords, {"extent": [90, 145, -10, 10]}],
                 [constrain_natural_earth, {"Country": ["Indonesia"]}]]
    regions = "Indonesia"
    
    make_nrt_inputs(data_dir, shapeinfo, regions)

    Europe_counties = ["Norway", "France", "Sweden", "Belarus", "Ukraine", "Poland", "Austria", "Hungary", "Moldova", "Romania", "Lithuania", "Latvia", "Estonia", "Germany", "Bulgaria", "Greece", "Albania", "Croatia", "Switzerland", "Luxembourg", "Belgium", "Netherlands", "Portugal", "Spain", "Ireland", "Italy", "Denmark", "United Kingdom", "Iceland", "Slovenia", "Finland", "Slovakia", "Czechia", "Bosnia and Herz", "North Macedonia", "Serbia", "Montenegro", "Kosovo"]
    shapeinfo = [[contrain_coords, {"extent": [-25, 60, 32, 73]}],
                 [constrain_natural_earth, {"Country": Europe_counties}]]
    regions = 'Europe'
    
    make_nrt_inputs(data_dir, shapeinfo, regions)

    shapefile = "/Focal_regions/SoW2526_Focal_MASTER_20260218.shp"    
    shapeinfo = [[contrain_to_sow_shapefile, {shp_filename:  data_dir + shapefile}]]
    region_names = [
                    "Midwestern Canadian Shield forests",
                    "Northwest Iberia"
                    ]
    make_nrt_inputs(data_dir, shapefile_path, regions)
    #run_era5_download_for_report(shapefile_path, region_names,  out_dir)
    #make_era5_extra_vars(data_dir + '/nrt_raw/', region_names)
    #browser()
    #temp_dir = data_dir + "/hadgem_nrt2/"
    #
    #for start_year in start_years:
    #    
    #    HadGEM_extract_variables(experiments, variables, start_year, dir, sub_dir,
    #                      out_dir, temp_dir,
    #                      region_names = region_names,
    #                      shapefile_path = shapefile_path)
    #    set_trace()
    
    #regrid_hyde_for_regions(region_names, data_dir) 
    
