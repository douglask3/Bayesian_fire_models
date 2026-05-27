library(raster)

fact_path = "data/data/driving_data_base/Pantanal/nrt/era5_monthly/"
cfact_path = "data/data/driving_data_base/Pantanal/nrt/era5_monthly/CF/"

files = c("dry_days", "max_consec_dry", "precip", "hurs_mean", "hurs_min", 
          "tas_mean", "tas_max")

BA_file = 'Fire_fraction_Pantanal'

BA_file_out = 'Fire_fraction_Pantanal-masked'

cf_files = list.files(cfact_path, full.names = TRUE, recursive = TRUE)
cf_files = cf_files[apply(sapply(files, grepl, cf_files), 1, any)]
open_and_check <- function(file, dir = NULL) {
    if (!is.null(dir)) file = paste0(dir,'/', file, '.nc')
    dat = brick(file)
    dat[dat>9E9] = NaN
    return(1-mean(is.na(dat)))
}

check_enemble <- function(i) {
    id = as.character(i)
    fileID = sapply(cf_files, function(i) substr(i, nchar(i)-5, nchar(i)-3))
    fnames = cf_files[which(as.numeric(gsub("\\D", "", fileID)) == i)]
    cfs = lapply(fnames, open_and_check)
    cf_mask = mean(do.call(addLayer, cfs)) == 1
    return(cf_mask)
}
#fact = open_and_check(BA_file, fact_path)
fact = lapply(files, open_and_check, fact_path)
fact_mask = mean(do.call(addLayer, fact)) == 1

i = 1

cf_mask = lapply(1:25, check_enemble)
cf_mask = mean(do.call(addLayer, fact)) == 1
BA = open_and_check(BA_file,fact_path)



mask = fact_mask & cf_mask
BA = raster::resample(BA, mask)
BA = brick(paste0(fact_path, BA_file, ".nc"))
BA[!mask] = NaN

writeRaster(BA, paste0(fact_path, BA_file_out, ".nc"), overwrite = TRUE)
browser()


#dats = check_variable(files[1])

