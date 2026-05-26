import iris
import numpy as np
import pandas as pd

# Load the cube
variables = ["hursmin", "pr", "sfcWind", "tas", "tasmax"]

for var in variables:
    cube = iris.load_cube("data/data/driving_data2425/nrt_attribution/Amazon_and_Rio_Negro_rivers/HadGEM_ALL/" + var + "/r001i1p3-2013-2.nc")

    # Convert time coordinate to datetime objects
    time_coord = cube.coord('time')
    time_datetimes = time_coord.units.num2date(time_coord.points)

    # Find indices where date is 1st January
    jan_1_indices = [(i, dt) for i, dt in enumerate(time_datetimes) if dt.month == 1 and dt.day == 1]

    # Turn into a DataFrame for a clean table
    df = pd.DataFrame(jan_1_indices, columns=['Time Index', 'Date'])
    df.to_csv("test-code/jan_layers" + var + ".csv")
    print(df)
