library(terra)
library(gdalUtils)

source("libs/add_date_time.r")


#path = '../temp/MODIS/'
temp_path = '../temp/regrid_vcf/'
output_path = '../data/data/driving_data/'

newproj = "+proj=longlat +datum=WGS84"
example_file = rast('../data/wwf_terr_ecos_0p5.nc')
extents = list(#c(-180.0, 180.0, -90.0, 90.0)
               c(-77.5, -56.0, -10.0, 2.0),
               c(-12.0, 49.0, 3.0, 63.0),
               c(-20.0, 50.0, -55.0, 55.0),
               #c(-85.0, -60.0, -30.0, 15.0),
               c(17.50, 33.0, 30.0, 43.0),
               c(-82.0, -55.0, -55.0, -5.0))
#extent = c(-170.0, -30.0, 30, 85)
area_names = c(#'Global',
              'SouthAmerica-box',
              'United_Kingdon',
              'MED',
              #'SouthAmerica',
              'Greece',
              'BoliviaChile')
paths = c('../temp/SouthAmerica-MODIS/', '../temp/UK-MODIS/', '../temp/MED-MODIS/', 
#'../temp/SouthAmerica-MODIS/', 
'../temp/Greece-MODIS/', '../temp/BoliviaChile-MODIS/')

#area_name = 'Canada'
variables = c("tree" = 1, "nontree" = 2, "nonveg" = 3)

forRegion <- function(area_name, extent, path) {
    eg_raster = rast(example_file)
    eg_raster = crop(eg_raster, extent)
    
    test_if_overlap <- function(r1, r2) {
        ext1 = ext(r1)
        ext2 = ext(r2)
        extc = intersect(ext1, ext2)
        return((extc[2] > extc[1]) & (extc[4] > extc[3]))
    }

    temp_path = paste0(temp_path, '/', area_name, '/') 
    dir.create(temp_path, recursive = TRUE) 
    files = list.files(path, full.name = TRUE)
    
    regrid_file <- function(file, band = 1, name = 'tree') {
        print(file)
    
        out_info0 = gsub('/', '', strsplit(file, path)[[1]][2], fixed = TRUE)
        temp_path = paste0(temp_path, '/', name, '/')
        dir.create(temp_path, recursive = TRUE) 
    
        out_info = paste0(temp_path, '-', band, '-', gsub('.hdf', '.txt', out_info0, fixed = TRUE))
        if (file.exists(out_info)) {
            info = read.table(out_info)[1,1]
            if (info == 'NoOverlap') return(NULL)
            return(rast(as.character(info)))
        }
    
        dat = rast(file, band)
    
        if (!all(extent == c(-180, 180, -90, 90))) {
            test = project(aggregate(dat, 100), newproj)
            overlap = test_if_overlap(test, eg_raster)
        } else overlap = TRUE
    
        if (!overlap || length(unique(dat)) == 1) {
            writeLines('NoOverlap', out_info)
            return(NULL)
        }
    
        dat = terra::project(dat, newproj)
    
        out_raster = crop(eg_raster, ext(dat) + c(-0.5, 0.5, -0.5, 0.5))
    
        find_area <- function(dat, ...) {
            #dat = aggregate(dat, 4)
            #dat = aggregate(dat, 0.5/rev(res(dat)), ...)
            #if ((ext(test)[2] - ext(test)[1])>180)  browser()
            dat = resample(dat, out_raster, ...)
        }
        veg_cover = land_cover = dat
        veg_cover[veg_cover>150] = 0
        veg_cover = find_area(veg_cover)
        if (all(is.na(veg_cover[]))) {
            writeLines('NoOverlap', out_info)
            return(NULL)
        }
        land_cover[land_cover < 150] = 1
        land_cover[land_cover > 150] = 0
        land_cover = find_area(land_cover, 'sum')
        out = c(veg_cover, land_cover)
    
        nc_out = paste0(temp_path, sub('.hdf', '.nc',out_info0, fixed = TRUE))
    
        writeCDF(out, nc_out, overwrite = TRUE)
        writeLines(nc_out, out_info) 
        return(out)
    }
    
    years = sapply(files, function(file) substr(strsplit(file, 'MOD44B.A')[[1]][2], 1, 4))
    mn = 3
    day = 6
    forVegType <- function(band, name) {
        print(band)
        print(name)
        print(area_name)
        output_path = paste0(output_path, area_name, '_extended/isimp3a')
        dir.create(output_path, recursive = TRUE) 
        output_fname = paste0(output_path, '/', name, '_raw.nc')
        temp_fname = paste0(temp_path, '/', name, '/')
        dir.create(temp_fname, recursive = TRUE) 
        temp_fname = paste0(temp_fname, '_collated.nc')
        
        if (file.exists(output_fname)) {
            out = rast(output_fname)
            if (length(unique(years)) == nlyr(out)) return(out)
        }
        
        dats = lapply(files, regrid_file, band, name)
        years = as.numeric(years)
     
        test = !sapply(dats, is.null)
        dats = dats[test]
        years = years[test]
    
        yearI = sort(unique(years))
        eg_raster[] = 0
        output = areaR = rep(eg_raster, max(2, length(yearI)))
        
        for (i in 1:length(dats)) {
            print(i)
            
            dat = resample(dats[[i]], eg_raster)
            dat[is.na(dat)] = 0
            whichY = which(yearI == years[[i]])
    
        
            output[[whichY]] = output[[whichY]] + dat[[1]] * dat[[2]]
        
            areaR[[whichY]] = areaR[[whichY]] + dat[[2]]
        }
        
        cover = output/areaR
        writeCDF(cover, temp_fname, overwrite=TRUE)
        mn = rep(mn, length(yearI))
        day = rep(day, length(yearI))
        day = day-(4*as.integer(yearI/4) == yearI)
        add_date_to_file(temp_fname, output_fname, yearI, mn, day, name, 
                         paste0(name, '-raw'), unit = '%')  
           
    }

    outs = mapply(forVegType, variables, names(variables))
     
    load_correct <- function(file) {
        out = rast(file)
        out = out[[nlyr(out)]]
        return(resample(out, eg_raster))
    }

    correct_files = paste0("../../fireMIPbenchmarking/data/benchmarkData/", 
                           c("treecover2000-2014.nc", "nontree2000-2014.nc", 
                            "bareground2000-2014.nc"))
    target = lapply(correct_files, load_correct)
    
    find_2014 = which(substr(time(outs[[1]]), 1, 4) == "2014")

    logit <- function(r, scale = 0.000000001) {
        mv = 1 - 2 *  scale
        r = r*mv + scale
        log(r/(-r + 1))
    }

    logistic <- function(r) 
        1/(1+exp(-r))

    correct_out <- function(r1, r2) {
        r1[r1>100] = 100
        r1 = r1 / 100
        r2 = r2 / 100
        r1t = logit(r1); r2t = logit(r2)
        out = r1t - r1t[[find_2014]] + r2t
        out = logistic(out)
        return(out)
    }

    corrected = mapply(correct_out, outs, target)
    corrected = lapply(corrected, '/', corrected[[1]] + corrected[[2]] + corrected[[3]])

    write_corrected <- function(rc, r0) {
        filename = sources(r0)[1]
        filename = gsub('_raw.nc', '_cover.nc', filename)
        
        varnames(rc) = gsub('_raw', '_cover', varnames(r0))
        units(rc) = units(r0)
        writeCDF(rc, filename, overwrite = TRUE)
    }
    
    mapply(write_corrected, corrected, outs)
}

mapply(forRegion, area_names, extents, paths)
