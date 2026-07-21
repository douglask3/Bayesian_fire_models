import iris
import numpy as np
import datetime
import cf_units
from pdb import set_trace

def fix_joeys_weird_time_coords(cube, file):
    #cube = cubes[-2]
    time_coord = cube.coord('time')
    
    def parse_weird_time(val, mn):
        # Split integer and fractional part
        date_int = int(val)
        frac = val - date_int
        
        # Parse YYYYMMDD
        try:
            dt = datetime.datetime.strptime(str(date_int), "%Y%m%d")
        except:
            year = int('20' + file.split('20')[1].split('_')[0])
            dt = datetime.datetime(year, mn+1, 15, 0, 0)
            frac = 0.5
        # Add fractional day (0.5 → 12:00)
        dt += datetime.timedelta(days=frac)
    
        return dt


    # Convert all points
    datetimes = np.array([parse_weird_time(v, mn) for mn, v in enumerate(time_coord.points)])
    
    # Define a proper unit
    unit = cf_units.Unit('hours since 1970-01-01 00:00:00', calendar='proleptic_gregorian')
    
    # Convert to numeric time
    numeric_time = unit.date2num(datetimes)
    
    # Build new coordinate
    new_time = iris.coords.DimCoord(
        numeric_time,
        standard_name='time',
        units=unit
    )
    # Replace coordinate
    cube.remove_coord('time')
    cube.add_dim_coord(new_time, 0)  # adjust dim if needed
    return cube



def add_years_onto_time(cube, coord_name='time', nyears = 1):
    # Access the time coordinate
    time_coord = cube.coord(coord_name)
    
    # 1. Convert numeric points to datetime objects
    dates = time_coord.units.num2date(time_coord.points)
    
    # 2. Add one year, handling leap year (Feb 29 -> Feb 28)
    new_dates = []
    for dt in dates:
        try:
            new_dates.append(dt.replace(year=dt.year + nyears))
        except ValueError:
            new_dates.append(dt.replace(year=dt.year + nyears, day=28))
            
    # 3. Convert updated dates back to numeric points
    new_points = time_coord.units.date2num(new_dates)
    
    # 4. Create a new coordinate and replace it in the cube
    new_time_coord = time_coord.copy(points=new_points)
    cube.replace_coord(new_time_coord)
    return cube
