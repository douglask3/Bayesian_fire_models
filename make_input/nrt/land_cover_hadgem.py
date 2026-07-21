from pathlib import Path

import iris
import iris.analysis

import cftime
import cf_units

import sys
sys.path.append('../../libs/')
sys.path.append('libs/')
from constrain_cubes_standard import *
from pdb import set_trace

factual_file = "/data/users/opatt/veg_2020_ancil/qrparm.HYDE31_2.pp"
counter_file = "/data/users/andrew.ciavarella/upgrade_ga6/veg/n216/hydra_ancil/qrparm.HYDE31_n216_1850.pp"

tree_frac = [0, 1]
grass_frac = [2, 3]
shrub_frac = [4, 5]
bare_frac = [7]

def for_veg_cube(cube, index, name):
    out = cube[index[0]].copy()
    if len(index) > 1:
        for i in index[1:]:
            out.data += cube.data[i]
    out.rename(name)
    return(out)
    set_trace()

def make_veg_cover(file, out_dir, experiment):
    cubes = iris.load(file)[-1]

    tree = for_veg_cube(cubes, tree_frac, "Tree_cover")
    shrub =  for_veg_cube(cubes, shrub_frac, "Shrub_cover")
    grass =  for_veg_cube(cubes, grass_frac, "Grass_cover")
    bare =  for_veg_cube(cubes, bare_frac, "Bare_cover")

    total = tree + shrub + grass + bare
    tree.data /= total.data
    shrub.data /= total.data
    grass.data /= total.data
    
    wood = tree.copy()
    wood.data += shrub.data
    wood.rename('Wood_cover')

    veg = wood.copy()
    veg.data += grass.data
    veg.rename('Veg_cover')

    veg_log = veg.copy()
    veg_log.data = np.log(1E-100 + veg_log.data)
    out_dir =  out_dir + experiment + '/'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    iris.save(tree, out_dir + 'tree.nc')
    iris.save(shrub, out_dir + 'shrub.nc')
    iris.save(grass, out_dir + 'grass.nc')
    iris.save(wood, out_dir + 'wood.nc')
    iris.save(veg, out_dir + 'veg_abs.nc')
    iris.save(veg_log, out_dir + 'veg_log.nc')
    return tree, wood, veg


if __name__=="__main__":
    dir_out = 'data/data/HadGEM_land_frac/'
    ftree, fwood, fveg = make_veg_cover(factual_file, dir_out, 'factual')
    ctree, cwood, cveg = make_veg_cover(counter_file, dir_out, 'counterfactual')
    
    
