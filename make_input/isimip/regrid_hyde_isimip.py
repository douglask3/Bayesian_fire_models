import sys
sys.path.append("make_input/nrt/")
from regrid_hyde import *

if __name__=="__main__":   
    exp_dirs = [
                "/isimp3a/counterclim/GSWP3-W5E5/period_1901_1920/",
                "/isimp3a/counterclim/GSWP3-W5E5/period_2000_2019/",
                #"/isimp3a/counterclim/GSWP3-W5E5/period_2002_2019/",
                "/isimp3a/obsclim/GSWP3-W5E5/period_2000_2019/",
                #"/isimp3a/obsclim/GSWP3-W5E5/period_2002_2019/",
                "/isimp3a/obsclim/GSWP3-W5E5/period_1901_1920/"]
    target_file = 'pr_mean.nc'
    for exp_dir in exp_dirs:
        regrid_hyde_all(regions[1:], hyde_files, exp_dir, target_file, hyde_dir)

