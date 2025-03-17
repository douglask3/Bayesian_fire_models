library(terra)
source("libs/add_date_time.r")

dir = '../data/data/burnt_area/'
files = list.files(dir, full.names = TRUE)
eg_raster = rast('../data/wwf_terr_ecos_0p5.nc')
res = 0.5
temp_out = '../temp/burnt_area_no_meta.nc'
file_out = '../data/data/driving_data/burnt_area.nc'

dat = rast(files)

ext(dat) = c(-180, 180, -90, 90)
fact = res/res(dat)
dat = aggregate(dat, fact)
dat = dat * fact[1] * fact[2] * 100000/cellSize(dat[[1]])

writeCDF(dat, temp_out, overwrite=TRUE)

dates = substr(sapply(files, function(i) strsplit(i,'C6_')[[1]][2]), 1, 6)

years = as.numeric(substr(dates, 1, 4))
mnns = as.numeric(substr(dates, 5, 6))
day = 15
add_date_to_file(temp_out, file_out, years, mnns, day, name = 'Burnt area')
