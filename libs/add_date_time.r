library(ncdf4)
library(terra)
library(raster)
add_date_to_file <- function(file, file_out = file, years, mnn, day, name = NULL, longname = NULL, overwrite_date = FALSE, ...) {
    
    print(file)
    if (!grepl('.nc', file)) return()
    
    rtest = raster(file)
        
    if (!overwrite_date & (substr(getZ(rtest) , 5, 5) == '-' && substr(getZ(rtest) , 8, 8) == '-')) return()

    dat = dat0 = rast(file) 
    
    #by = length(years)/nlayers(dat)
    #yr = floor(seq(st, length.out = nlayers(dat), by = by))
    
    #mnn = rep(1:12, length.out = nlayers(dat))
    mn = as.character(mnn)
    mn[mnn < 10] = paste0('0', mn[mnn < 10])
    
    dy = as.character(day)
    dy[day < 10] = paste0('0', dy[day < 10])
    date = as.Date(paste(years, mn, dy, sep = '-'))
    
    names(dat) = date
    time(dat) = date
    #dat = setZ(dat, as.Date(date), 'Date')
    if (!is.null(name)) varnames(dat) = name
    if (!is.null(longname)) longnames(dat) = longname
    
    writeCDF(dat, file_out, overwrite = TRUE, ...)
}
