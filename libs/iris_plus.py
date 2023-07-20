
import cf_units
import iris
from pdb import set_trace


#Function to sort out the time dimension
def sort_time(cube, field, filename):
    
    cube.coord("time").bounds=None
    tcoord = cube.coord("time")
    tcoord.units = cf_units.Unit(tcoord.units.origin, calendar="gregorian")
    tcoord.convert_units("days since 1661-01-01 00:00:00")
    tcoord.units = cf_units.Unit(tcoord.units.origin, calendar="proleptic_gregorian")
    cube.remove_coord("time")
    cube.add_dim_coord(tcoord, 0) # might need to find this dimension

    try:
        iris.coord_categorisation.add_year(cube, 'time')
    except:
        pass

    
    try:               
        cube.remove_coord("month")
        iris.coord_categorisation.add_month_number(cube, 'time', name='month')
    except:
        pass

    try:
        del cube.attributes["history"]
    except:
        pass
    return(cube)


def add_bounds(cube):
    coords = ('time', 'longitude', 'latitude')
    for coord in coords:
        try: 
            cube.coord(coord).guess_bounds()
        except:
            pass
            #if not cube.coord(coord).has_bounds():
            
    return(cube)


def insert_sim_into_cube(x, eg_cube, mask):
    Pred = eg_cube.copy()
    pred = Pred.data.copy().flatten()

    try:        
        pred[mask] = x
    except:
        set_trace()
    Pred.data = pred.reshape(Pred.data.shape)
    return(Pred)

