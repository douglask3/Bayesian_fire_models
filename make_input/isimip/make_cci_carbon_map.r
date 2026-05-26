library(raster)
library(ncdf4)
source("cfg.r")


dir_in = 'data/carboncci/'

dir_our = 'outputs/carboncci/'

files = list.files(dir_in)

ncuts = 72
nboots = 10

egRaster = raster('data/GFED4s_burnt_area.nc')
egRaster[] = 0.0
egArea = raster::area(egRaster)
tfile0 = 'temp/make_cci_carbon_map-2-cut'

write_temp_text_file <- function(tfile, txt) {
    if (file.exists(tfile)) unlink(tfile)
    fileConn = file(tfile)
    writeLines(c(txt), fileConn)
    close(fileConn)
}

regridFile <- function(file) {
    file_in = paste0(dir_in, file)
    fileID =  substr(file, 1, nchar(file)- 3)
    if (grepl("CHANGE", file)) return()
    nc = nc_open(file_in)
    variables = names(nc$var)[1:2]
    
    mu = raster(file_in, varname = variables[1])
    sigma = raster(file_in, varname = variables[2])

    extent = extent(egRaster)

    cut_extents <- function(i, j) {
        out = seq(extent[i], extent[j], length.out = ncuts+1)
        return(head(out, -1))
    }
    xs = cut_extents(1, 2); ys = cut_extents(3, 4)
    
    xdiff = diff(xs[1:2]); ydiff = diff(ys[1:2])
    regrid4cut <- function(x, y) {
        
        tfile = paste(tfile0, ncuts, x, y, xdiff, ydiff,
                      fileID, '.txt', sep = '-')
        print(file_in)
        print(x)
        print(y)
        tfile_nc = paste(tfile0, 'nc', ncuts, nboots, x, y, xdiff, ydiff,
                         fileID, '.nc', sep = '-')
        #tfile_nc_area = paste(tfile0, 'nc', ncuts, nboots, x, y, xdiff, ydiff,
        #                 fileID, '-area.nc', sep = '-')
        
        nboots1 = nboots
        outs0 = NULL
        if (file.exists(tfile)) {
            info = read.table(tfile, stringsAsFactors=FALSE)
            if (info[1,1] == "empty") {
                return()        
            } else {
                if (file.exists(tfile_nc)) {
                    return(brick(tfile_nc))
                } else {
                    tfile0 = file.exists(info[1,1])
                    if (tfile0) {
                        outs0 = brick(info[1,1])
                        nboots0 = nlayers(outs0) - 1
                        if (nboots0 > nboots) {
                            return(outs0[[1:nboots]])
                        } else {
                            nboots1 = nboots - nboots0
                        }
                    }
                }
            }
        }
        
        cut_extent = c(x, x+xdiff, y, y+ydiff)  
        
        mu = try(terra::crop(mu, cut_extent), silent = TRUE)
        if (class(mu) == "try-error" || is.na(maxValue(mu)) || maxValue(mu) == 0) {
            write_temp_text_file(tfile, "empty")
            return()
        }
        egRasteri = raster::crop(egRaster, cut_extent)
        
        rarea = mu
        rarea[!is.na(rarea)] = 1
        rarea = terra::extend(rarea, egRasteri)
        rarea[is.na(rarea)] = 0
        rarea = terra::resample(rarea, egRasteri) 
        #rarea = writeRaster(rarea, file = tfile_nc_area, overwrite = TRUE)
        
        mu_agg = terra::resample(mu, egRasteri) 
        sigma = terra::crop(sigma, cut_extent)
        
        print("yay")
        doBoot <- function(id) {
            print(id)
            mu_boot = sigma* rnorm(ncell(sigma), 0, 1)            
            boot = mu_agg + terra::resample(mu_boot, egRasteri) 
            boot[boot<0] = 0
            return(boot)
        }
        out = layer.apply(1:nboots1, doBoot)
        out = addLayer(rarea, out)
        if (!is.null(outs0))out = addLayer(out, outs0[[-1]])
        out = writeRaster(out, file = tfile_nc, overwrite = TRUE)
        write_temp_text_file(tfile, c(tfile_nc, nboots))
        if (!is.null(outs0)) unlink(info[1,1])
        return(out)        
    }

    
    combine_all_ens <- function(outs, resam = TRUE, byArea = TRUE) {

        selectResample <- function(out, i, byArea = TRUE) {
            
            if (byArea) rarea = out[[1]]
            out = out[[min(nlayers(out), i)]] 
            if (byArea) out = out * rarea
            if (resam) out = raster::resample(out, egRaster)
            out
        }
        combine_ens <- function(i, ...) {
            print(i)
            ens = mapply(selectResample, outs, i, ...)
            ens = layer.apply(ens, function(i) i)
            sum(ens, na.rm = TRUE)
        }
        rarea = combine_ens(1, byArea = FALSE)
        out = layer.apply(2:max(sapply(outs, nlayers)), combine_ens, byArea = byArea)
        out = addLayer(rarea, out)
        return(out)
        
    }

    nn <<- 0
    applyOverY <- function(y) {        
        tfile = paste(tfile0, ncuts, nboots,'ymerge', y, fileID, '.nc', sep = '-')
        print(tfile)
        
        if (file.exists(tfile)) return( brick(tfile))
        #return()
        lock_file = paste(tfile0, 'lock', y, fileID, '.txt', sep = '-')
        
        if (file.exists(lock_file)) return('locked')
        file.create(lock_file)
       
        out  = lapply(xs, regrid4cut, y)
        out = unlist(out)
        
        if (is.null(out))  out = addLayer(egRaster, egRaster)
            else out = combine_all_ens(out)
        
        writeRaster(out, file = tfile, overwrite = TRUE)
        
        return(out)
    }
    
    file_out = paste0('outputs/',
                      paste(strsplit(fileID, '100m')[[1]], collapse = '0.5Degree'), 
                      '-ensembles-', nboots, '.nc')
    
    if (file.exists(file_out) && FALSE) return(brick(file_out))
    outs = lapply(rev(ys), applyOverY)
    
    if (any(sapply(outs, function(i) is.character(i)))) stop("some locked files")
    outs = outs0 = outs[!sapply(outs, is.null)]
    
    outs = combine_all_ens(outs, FALSE, FALSE)
    mask = raster('/scratch/hadea/isimip3a/u-cc669_isimip3a_fire/20CRv3-ERA5_obsclim/jules-vn6p3_20crv3-era5_obsclim_histsoc_default_burntarea-total_global_monthly_1901_2021.nc')
    mask = (sum(outs[[-1]])==0) & (is.na(mask))
    outs = layer.apply(outs, function(r) {r[mask] = NaN; r})
    writeRaster(outs, file = file_out, overwrite = TRUE)  
}

outs = lapply(files, regridFile)
outs = outs[-1]

Dboots = 100


r1 = sample(2:nlayers(outs[[1]]), Dboots, replace = TRUE)
r2 = sample(2:nlayers(outs[[2]]), Dboots, replace = TRUE)

dout = outs[[2]][[r2]] - outs[[1]][[r1]]
dout = addLayer(sqrt(outs[[1]][[1]] * outs[[2]][[1]]), dout)
findYr <- function(i)
    strsplit(strsplit(filename(outs[[i]]), 'Degree-')[[1]][2], '-')[[1]][1]

yrs = sapply(2:1, findYr)

file_out = paste0(strsplit(filename(outs[[1]]), yrs[2])[[1]], 
                  collapse = paste0(c('CHANGE', yrs), collapse = '-' ))

writeRaster(dout, file = file_out, overwrite = TRUE)
