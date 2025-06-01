Amazon_long_name = "Amazon forest northeast of the Amazon and Rio Negro rivers"
LA_long_name = "Mediterranean portion of Los Angeles, Orange County, Ventura, San Bernardino, Riverside, San Diego"

regions_2324 = {"Congo"   : {"longname": "Central Congo basin forests", 
                             "shortname": "Congo Basin", 
                             "years": [2024], "mnths":  ['07']},
                "Amazon"  : {"longname": Amazon_long_name, 
                             "shortname":  "Northeast Amazon", 
                             "years": [2024], "mnths":  ['01', '02', '03']},
                "LA"      : {"longname": LA_long_name, 
                             "shortname":  "Southern California", 
                             "years": [2025], "mnths":  ['01']},
                "Pantanal": {"longname": "Greater Pantanal basin plus Chiquitano forests", 
                             "shortname":  "Pantanal and Chiquitano",
                             "years": [2024], "mnths":  ['08', '09']},
                "Alberta" : {"longname": "Alberta Mountain forests including Jasper NP", 
                             "shortname":  "Alberta", 
                             "years": [2024], "mnths":  ['07']},
                "NEIndia" : {"longname": "Forests of northeast India, Nepal, Bangladesh", 
                             "shortname":  "Himalayan Foothills ", 
                             "years": [2024], "mnths":  ['04']}}

def get_region_info(region_names, regions_info = regions_2324):
    if ~isinstance(region_names, list): region_names = [region_names]
    return {key: regions_info[key] for key in region_names if key in regions_info}
