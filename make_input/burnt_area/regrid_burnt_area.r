library(terra)
source("libs/add_date_time.r")

dir = 'data/data/burnt_area/modis0p05/'
files = list.files(dir, full.names = TRUE)
eg_raster = rast('data/wwf_terr_ecos_0p5.nc')

res = 0.05
temp_out = 'temp2/burnt_area_no_meta.nc'
#file_out = 'data/data/driving_data2425/burnt_area0p05.nc'

# for global set to NULL
shape_file = "data/data/SoW2425_shapes/SoW2425_Focal_MASTER_20250221.shp"
shape_names = list("northeast India",
                   "Alberta",
                   "Los Angeles",
                   "Congo basin",
                   "Amazon and Rio Negro rivers",
                   "Pantanal basin")

region_names = c('NEIndia', 'Alberta', 'LA', 'Congo', 'Amazon', 'Pantanal')
eg_raster_dir  = "data/data/driving_data2425/<<REGION>>/isimp3a/obsclim/GSWP3-W5E5/period_2002_2019/tree_cover_jules-es.nc"
eg_raster_nrt_dir = 'data/data/driving_data2425/<<REGION>>/nrt/era5_monthly/precip.nc'
out_file_hires  = "data/data/driving_data2425/<<REGION>>/burnt_area.nc"
out_file_nrt    = "data/data/driving_data2425/<<REGION>>/nrt/era5_monthly/burnt_area.nc"
out_file_isimip = "data/data/driving_data2425/<<REGION>>/isimp3a/obsclim/GSWP3-W5E5/period_2002_2019/burnt_area.nc"
dat = rast(files)
ext(dat) = c(-180, 180, -90, 90)
fact = res/res(dat)

shp = vect(shape_file)


if (any(fact) != 1) {
    dat = aggregate(dat, fact)
    dat = dat * fact[1] * fact[2] * 100000/cellSize(dat[[1]])
}

for_region <- function(rname, sname) {
    
    eg_raster       = rast(gsub('<<REGION>>', rname, eg_raster_dir))
    eg_raster_nrt   = rast(gsub('<<REGION>>', rname, eg_raster_nrt_dir))
    out_file_hires  = gsub('<<REGION>>', rname, out_file_hires )
    out_file_nrt    = gsub('<<REGION>>', rname, out_file_nrt   )
    out_file_isimip = gsub('<<REGION>>', rname, out_file_isimip)

    shp_rgn = shp[grep(sname, shp$name, ignore.case = TRUE), ]
    extent = ext(shp_rgn)
    extent[3:4] =  -extent[4:3]
    dat = crop(dat, extent)
    dat0 = dat
    dat = flip(dat, 'vertical')
    extent[3:4] =  -extent[4:3]
    ext(dat) = extent
    dat = mask(dat, shp_rgn)  

    dates = substr(sapply(files, function(i) strsplit(i,'C6_')[[1]][2]), 1, 6)
    years = as.numeric(substr(dates, 1, 4))
    mnns = as.numeric(substr(dates, 5, 6))
    day = 15
    write_out <- function(dat, file_out, which_dates = 1:length(years)) {
        writeCDF(dat, temp_out, overwrite=TRUE)
        add_date_to_file(temp_out, file_out, 
                         years[which_dates], mnns[which_dates], day, 
                         name = 'Burnt area')
    }
    
    write_out(dat, out_file_hires)
    
    #writeCDF(dat, temp_out, overwrite=TRUE) 
    dat_nrt = terra::resample(dat, eg_raster_nrt[[1]])
    dat_nrt = mask(dat_nrt, eg_raster_nrt[[1]])
    
    yr_range = range(as.numeric(substr(time(eg_raster_nrt), 1, 4)))
    test_years = years > yr_range[1] & years < yr_range[2]
    write_out(dat_nrt[[test_years]], out_file_nrt, test_years)

    dat = terra::resample(dat, eg_raster[[1]])
    dat = mask(dat, eg_raster[[1]])
    #write_out(dat, out_file_nrt)

    yr_range = range(as.numeric(substr(time(eg_raster), 1, 4)))
    test_years = years > yr_range[1] & years < yr_range[2]
    dat = dat[[test_years]]
    #years = years[test_years]
    #mnns = mnns[test_years]
    
    write_out(dat, out_file_isimip, test_years)
}

mapply(for_region, region_names, shape_names)
