source("../gitProjectExtras/gitBasedProjects/R/sourceAllLibs.r")
sourceAllLibs('../rasterextrafuns/rasterPlotFunctions/R/')
source("libs/return_multiple_from_functions.r")

library(plotrix)
library(mapdata)
library(mapplots)

library(rgdal)

#SA_ste <- readOGR(dsn = "data/South_America", layer = "South_America")
#rivers <- readOGR(dsn = "data/majorrivers_0_0", layer = "MajorRivers")

StandardLegend <- function(cols, limits, dat, rightx = 0.95, extend_max = TRUE, oneSideLabels = TRUE, add = FALSE, ...) {
    if (add)        
        plot_loc = c(0.41, rightx, 0.1, 0.13)
    else 
        plot_loc = c(0.01, rightx, 0.3, 0.56)
    add_raster_legend2(cols, limits, dat = dat, add = add,
                       transpose = FALSE, srt = 0, oneSideLabels= oneSideLabels,
                       plot_loc = plot_loc,
                       ylabposScling = 1, extend_max = extend_max, ...)
}

lineBox <- function(x, y, ...) 
    lines(c(x[1], x[2], x[2], x[1], x[1]), c(y[1], y[1], y[2], y[2], y[1]),...)

plotStandardMap <- function(r, cols, limits, e = NULL, add_legend = FALSE,
                            limits_error = c(0.2, 0.200000001),
                            title2 = '', title3 = '', xlim = c(-180, 180), ylim = c(-60, 90),
                            ePatternRes = 67, ePatternThick = 0.5,
                            ...,  add = FALSE, speedy = T) {
    

    if (nlayers(r) == 2) {
        e = r[[2]]
        r = r[[1]]
    }
    if (nlayers(r) > 1 && is.null(e)) {
        if (nlayers(r) == 3) {
            if (any(r[] <0, na.rm = TRUE) && any(r[] > 0, na.rm = TRUE)) {
                e = r[[2]]
                #e[!is.na(r[[2]])] = 0
                e = abs(r[[3]] - r[[1]])/max(abs(r[[c(1,3)]]))/2
                e[r[[3]]>0 & r[[1]] <0] = 1                
            } else  e = 1-r[[1]]/r[[3]]
            
            r = r[[2]]
        } else {
            e = sd.raster(r)
            r = mean(r)
        }
    } 
    if (!speedy) {
        r[r>9E9] = NaN
        if (!is.null(e)) e[is.na(r)] = NaN
        mask = raster('data/seamask.nc')
    
        r[mask != 2] = NaN
        if(!is.null(e)) e[mask != 2] = NaN
    }
    #plot(xlim, ylim, xlab = '', ylab = '', axes = FALSE, type ='n')
    
    if (speedy) coast.lwd = 1 else coast.lwd = NULL
    plot_raster_from_raster(r, e = e, coast.lwd = coast.lwd,
                            cols = cols, limits = limits, add_legend = FALSE,
                            quick = TRUE, ePatternRes = ePatternRes, ePatternThick = ePatternThick,
                            limits_error = limits_error, add = add, ...)
    
    #if (!speedy) addCoastlineAndIce2map()
    if (!speedy) for (i in 1:4) addCoastlineAndIce2map()
    
    if (add_legend) {
        add_raster_legend2(cols, limits, dat = r,
                           transpose = FALSE, srt = 0, oneSideLabels= FALSE,
                           plot_loc = c(0.35, 0.99, 0.09, 0.12),  ylabposScling=0.8, ...)
    }
    grid()
}

addCoastlineAndIce2map <- function() {
    add_icemask()
    
    mask = raster('data/seamask.nc')
    mask = mask>1
    
    plot_raster_from_raster(mask+1, add = TRUE, 
                             cols = c("white", "transparent"),readyCut = TRUE,
                             limits =  NULL, quick = TRUE, interior = FALSE, 
                             coast.lwd = NULL, add_legend = FALSE)
    #
    #contour(mask, add = TRUE, drawlabels = FALSE, lwd = 0.5)  

    ployBox <- function(x, y)
        polygon(c(x[1], x[2], x[2], x[1]), c(y[1], y[1], y[2], y[2]), col = "white", border = "white")
        
    ployBox(c(-180, -90), c(-60, 0))
    ployBox(c(-180, -120), c(-60, 25))
    ployBox(c(-50, -19), c(10, 25))
    ployBox(c(-50, -13.5), c(27.5, 34))
    ployBox(c(115, 125), c(-8, -7))
    ployBox(c(104, 111), c(2.5, 8))
    ployBox(c(122, 128), c(2.5, 5)) 
}

add_icemask <- function() {
	icemask = raster('data/icemask.nc')
	plot_raster_from_raster(icemask, add = TRUE, cols = c('#FFFFFFFF', 'grey'), y_range = c(-60, 90),
						    limits = c(-0.5, 0.5), add_legend = FALSE, interior = FALSE, coast.lwd = 0.67)#, coast.lwd = NULL)
}

