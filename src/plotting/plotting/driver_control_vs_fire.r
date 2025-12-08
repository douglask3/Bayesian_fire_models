library(raster)
library(interp)
source("libs/plot_raster_map.r")
source("libs/logistic.r")
source("libs/logit.r")

dir0 = 'outputs/outputs_scratch/ConFLAME_nrt-drivers6/<<region>>-2425/samples/_21-frac_points_0.5/baseline-/'

regions = c('Amazon', 'Pantanal', 'LA', 'Congo')
region_names = c('Northeast Amazon', 'Pantanal and Chiquitano', 'Southern California', 'Congo Basin')

dirs = c("Burned Area" = "Evaluate", "Fuel" = "Standard_0", "Moisture" = "Standard_1",
         "Weather" = "Standard_2",
         "Wind" = "Standard_3", "Ignitions" = "Standard_4", 
         "Suppression" = "Standard_5")


cols = c("#cfe9ff", "#fc6", "#f68373", "#c7384e", "#862976")


openDat <- function(dir, region, nfiles = 1000, layers = 253:255) {
    tfile = paste0(c('temp/driver_control_vs_fire_', dir, region, nfiles, layers, '.csv'), 
                   collapse = '-')
    
    if (file.exists(tfile)) 
        return(read.csv(tfile, stringsAsFactors = FALSE)[,-1])
    
    dir0 = gsub('<<region>>', region, dir0)

    
    files = list.files(paste0(dir0, dir, '/'), pattern = "-pred", full.names=TRUE)
    files = files[seq(1,length(files), length.out = nfiles)]
    
    open_file <- function(file) {
        print(file)
        out = brick(file)[[layers]]
        if (!exists('map_mask')) {
            map_mask = !is.na(sum(out)) & out[[1]] <9E9
            map_mask <<- map_mask
        }
        
        return(out[map_mask])
    }
    outs = sapply(files, open_file)
    write.csv(outs, file = tfile)
    return(outs)
}

plot_BA_vs_control <- function(i, dat, y0, y, region_name, plotLast = FALSE) {
    x = logit(unlist(dat[[i]]))
    
    z = 100*(-y0 + y0/unlist(dat[[i]]))
    x = x[y>-12]
    z = z[y > -12]
    y = y[y >-12]
    
    cols = cols[unlist(mapply(rep, 1:5, 9 + (1:5)^4))]
    Dplot <- function(xi, yi) {
        cols = densCols(xi,yi, colramp = colorRampPalette(cols), bandwidth = 0.1) 
        plot(yi~xi, pch = 20, col = cols, cex = 1, axes = FALSE)
    }
    set_axes_pos <- function(x, axis,
                             xlab = c(0, 0.1, 0.2, 0.5, 1, 10, 25, 50, 
                                      75, 90, 99, 99.5, 99.8, 99.9, 100)) {
        xlab = 100*unique(signif(logistic(seq(min(x), max(x), , 10)), 1))
        if (length(xlab) < 4) browser()
        
        pos = logit(xlab/100)
        axis(axis, at = pos, labels = xlab)
    }
    if (!plotLast) {
        Dplot(x, y)
        set_axes_pos(x, 1)
        mtext(side = 2, names(dirs)[i], line = 2)
        if (i == 2) mtext(side = 3, 'Control vs Burned Area (BA)')
        if (i == length(dirs)) mtext(side = 1, 'Control', line = 2)
        set_axes_pos(y, 2)
        Dplot(x, z)
        set_axes_pos(x, 1)
        if (i == 2) mtext(side = 3, 'Control vs Potential BA increase')
        if (i == length(dirs)) mtext(side = 1, 'Control', line = 2)
        axis(2)
    }
    Dplot(y, z)
    if (plotLast) {
        set_axes_pos(y, 1)
        #set_axes_pos(z, 2)
        axis(2)
        if (region_name == region_names[1]) mtext(side = 2, names(dirs)[i], line = 2)
        if (i == 2) mtext(side = 3, region_name)
        
    } else {
        set_axes_pos(y, 1)
        if (i == 2) mtext(side = 3, 'BA vs Potential BA increase')
        if (i == length(dirs)) mtext(side = 1, 'BA', line = 2)
        axis(2)
    }

}

plot_region_plots <- function(region, region_name, name, plotLast = FALSE) {
    rm(map_mask, envir = .GlobalEnv)
    dat = lapply(dirs, openDat, region)
    rm(map_mask, envir = .GlobalEnv)
    y0 = unlist(dat[[1]])
    y = logit(y0)

    if (!plotLast) {         
        png(paste0("figs/nrt-control-scatter", region, '-', name, ".png"), 
            height = height, width = width, 
            units = 'in', res = 300)
        par(mfrow = c(6, 3), mar = rep(1, 4), oma = c(3, 3, 2, 2))
    }
    
    sapply(2:7, plot_BA_vs_control, dat, y0, y, region_name, plotLast = plotLast)
    
    if (!plotLast) dev.off()
}
png("figs/nrt-control-scatter-potnetial.png", 
            height = 9.5, width = 9.5, 
            units = 'in', res = 300)
par(mfcol = c(6,4), mar = rep(1, 4), oma = c(3.5, 5, 2, 0))
mapply(plot_region_plots, regions, region_names, 'lastOnly', plotLast = TRUE )
mtext(side = 1, outer = TRUE, line = 1.5, 'Burned area (%)')
mtext(side = 2, outer = TRUE, line = 2.5, 'Potential Burned area Increase from control (%)')
dev.off()
browser()
