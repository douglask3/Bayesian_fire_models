
from pdb import set_trace
import pandas as pd
import numpy as np

def extract_years(df, years, mnths, ext = "-01T00:00:00"):
    """
    Extracts and averages values from a DataFrame across specified months and years.

    Parameters:
    ----------
    df : pandas.DataFrame
        A DataFrame with columns labeled by timestamp strings (e.g., '2023-01-01T00:00:00') 
        and rows representing ensemble members or samples.
    years : list of str or int, or None
        The years to include (e.g., [2023, 2024]). If None, all years found in column names are used.
    mnths : list of str
        The months to average over (e.g., ['01', '02', '03'] for January–March).
    ext : str, optional
        A string pattern representing the suffix to match in column names (default is '-01T00:00:00'), 
        though it's overridden in favor of a wildcard match.

    Returns:
    -------
    np.ndarray
        A 1D numpy array with the average value for each row (e.g., ensemble member), 
        computed across the selected months for each year. The result is flattened to combine years.
    
    Notes:
    -----
    - Column matching uses wildcards via `fnmatch` to allow flexibility in timestamp formats.
    - Designed for use in ensemble climate/fire datasets where time is encoded in column headers.
    - Useful for computing seasonal means (e.g., JFM) per year, per ensemble member.
    """
    if years is None:
        years = np.unique([col[0:4] for col in df.columns[1:]])
    # Reshape: group columns by year
    avg_per_year = []
    for year in years:
        cols_this_year = [
            col for col in df.columns
            for month in mnths
            if fnmatch.fnmatch(col, f"{year}-{month}*")
        ]
    
        avg_per_year.append(df[cols_this_year].mean(axis=1))
    
    return np.array(avg_per_year).flatten()

