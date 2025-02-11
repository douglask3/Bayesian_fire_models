library(terra)

path_in = "data/data/faPAR_and_drivers/climate/"

path_out = "data/data/faPAR_and_drivers/climate_regid/"

vars = c('pre',  'tmp',  'vpd', 'ppfd')

files = list.files(path_in, full.names = TRUE)

forVar <- function(var) {
    dat = rast(files, var)
    file_out = paste0(path_out, var, '.nc')
    writeCDF(dat, file_out, overwrite = TRUE)
}

lapply(vars, forVar)
