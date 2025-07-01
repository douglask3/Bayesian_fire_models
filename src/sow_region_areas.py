import geopandas as gp

shape_names = ["Congo basin", 
                   'northeast India', "Alberta",
                   "Los Angeles",
                   "Amazon and Rio Negro rivers",
                   "Pantanal"]
shp_filename = "data/data/SoW2425_shapes/SoW2425_Focal_MASTER_20250221.shp"

shp = gp.read_file(shp_filename)
shp["geometry"] = shp["geometry"].buffer(0)
for name in shape_names:
    # Project to suitable CRS
    shp_proj = shp.to_crs(shp.estimate_utm_crs())

    # Get geometry and area
    geom_proj = shp_proj[shp_proj['name'].str.contains(name, case=False, na=False)].geometry.unary_union
    area_km2 = geom_proj.area / 1e6

    print(f"For: {name}, Area: {area_km2:.2f} km²")

#try:
#        geom = shp[shp['name'].str.contains(name, case=False, na=False)].geometry.unary_union
#    except:
##        
#        geom = shp[shp['name'].str.contains(name, case=False, na=False)].geometry.unary_union

