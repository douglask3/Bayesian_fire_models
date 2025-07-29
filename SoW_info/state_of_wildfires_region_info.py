from pdb import set_trace
Amazon_long_name = "Amazon forest northeast of the Amazon and Rio Negro rivers"
LA_long_name = "Mediterranean portion of Los Angeles, Orange County, Ventura, San Bernardino, Riverside, San Diego"

# Dictionary defining region-specific metadata for the 2023–2024 fire seasons.
# Includes long and short display names, analysis years, and months of interest.
regions_2324 = {"Congo"   : {"longname": "Central Congo basin forests", 
                             "shortname": "Congo Basin", 
                             "years": [2024],  "mnths":  ['07'],
                             "Anomoly_levels": [-10, -5, -2, -1, 1, 2, 5, 10],
                             "Ratio_levels": [0.2, 0.25, 0.5, 1, 2, 4, 5],
                             "dir": "Congo"},
                "Amazon"  : {"longname": Amazon_long_name, 
                             "shortname":  "Northeast Amazonia", 
                             "years": [2024], "mnths":  ['01', '02', '03'],
                             "Anomoly_levels": [-10, -5, -2, -1, 1, 2, 5, 10],
                             "Ratio_levels": [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20],
                             "dir": "Amazon"},
                "LA"      : {"longname": LA_long_name, 
                             "shortname":  "Southern California", 
                             "years": [2025], "mnths":  ['01'],
                             "Anomoly_levels": [-10, -5, -2, -1, 1, 2, 5, 10],
                             "Ratio_levels": [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20],
                             "dir": "LA"},
                "Pantanal": {"longname": "Greater Pantanal basin plus Chiquitano forests", 
                             "shortname":  "Pantanal and Chiquitano",
                             "years": [2024], "mnths":  ['08', '09'],
                             "Anomoly_levels": [-20, -10, -5, -2, -1, 1, 2, 5, 10, 20],
                             "Ratio_levels": [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20],
                             "dir": "Pantanal"},
                "Alberta" : {"longname": "Alberta Mountain forests including Jasper NP", 
                             "shortname":  "Alberta", 
                             "years": [2024], "mnths":  ['07'],
                             "Anomoly_levels": [-10, -5, -2, -1, 1, 2, 5, 10],
                             "Ratio_levels": [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20],
                             "dir": "Alberta"},
                "NEIndia" : {"longname": "Forests of northeast India, Nepal, Bangladesh", 
                             "shortname":  "Himalayan Foothills ", 
                             "years": [2024], "mnths":  ['04'],
                             "Anomoly_levels": [-10, -5, -2, -1, 1, 2, 5, 10],
                             "Ratio_levels": [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20],
                             "dir": "NEIndia"}}

def get_region_info(region_names, regions_info = regions_2324):
    """
    Retrieve metadata for one or more regions from the regions_2324 dictionary.

    Parameters:
    ----------
    region_names : str or list of str
        Name(s) of region(s) to retrieve. If a single region is provided as a string,
        it will be converted into a list.
    regions_info : dict, optional
        Dictionary containing region metadata. Defaults to `regions_2324`.

    Returns:
    -------
    dict
        Dictionary of metadata for the requested region(s), including long name,
        short name, analysis years, and months.

    Example:
    --------
    #>>> get_region_info("Amazon")
     {'Amazon': {'longname': 'Amazon forest northeast of the Amazon and Rio Negro rivers',
                'shortname': 'Northeast Amazon',
                'years': [2024],
                'mnths': ['01', '02', '03']}}
    """
    
    if not isinstance(region_names, list): region_names = [region_names]
    
    return {key: regions_info[key] for key in region_names if key in regions_info}
