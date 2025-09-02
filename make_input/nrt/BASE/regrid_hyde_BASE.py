
import sys
sys.path.append('.')
sys.path.append('make_input/nrt/')
from regrid_hyde import *

if __name__=="__main__":
    regions = ["Amazon"]
    target_dir = "data/data/driving_data_base/"
    regrid_hyde_for_regions(regions, hyde_files, target_file, hyde_dir, target_dir)
