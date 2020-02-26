source("../gitProjectExtras/gitBasedProjects/R/sourceAllLibs.r")
sourceAllLibs('../rasterextrafuns/rasterPlotFunctions/R/')
source("libs/return_multiple_from_functions.r")

library(plotrix)
library(mapdata)
library(mapplots)

StandardLegend <- function(cols, limits, dat, rightx = 0.95, extend_max = TRUE, oneSideLabels = TRUE, ...) 
        add_raster_legend2(cols, limits, dat = dat, add = FALSE,
                           transpose = FALSE, srt = 0, oneSideLabels= oneSideLabels,
                           plot_loc = c(0.01, rightx, 0.3, 0.56),
                           ylabposScling = 1, extend_max = extend_max, ...)

lineBox <- function(x, y, ...) 
    lines(c(x[1], x[2], x[2], x[1], x[1]), c(y[1], y[1], y[2], y[2], y[1]),...)

plotStandardMap <- function(r, cols, limits, e = NULL, add_legend = FALSE,
                            limits_error = c(0.5, 0.500000001),
                            title2 = '', title3 = '', ...) {
    
    if (nlayers(r) > 1 && is.null(e)) {
        if (nlayers(r) == 3) {
            e = 1-r[[1]]/r[[3]]
            r = r[[2]]
            
        } else {
            e = sd.raster(r)
            r = mean(r)
        }
    } 
    r[r>9E9] = NaN
    if (!is.null(e)) e[is.na(r)] = NaN
    plot(c(-85, -36), c(-56, 15), xlab = '', ylab = '', axes = FALSE, type ='n')
    grid()
    plot_raster_from_raster(r, e = e,
                            cols = cols, limits = limits, add_legend = FALSE,
                            quick = TRUE, ePatternRes = 5, ePatternThick = 0.67,
                            limits_error = limits_error, add = TRUE, ...)

    
    #lineBox(-c(71.25, 63.75), -c(11.25,  6.25))     
    #lineBox(-c(61.25, 53.75), -c(11.25,  6.25))   
    #lineBox(-c(48.25, 43.25), -c( 8.75,  1.25))
    #lineBox(-c(66.25, 58.75), -c(18.75, 13.75))
               
    polygon(c(-62.5, -35, -35, -62.5), c(-56, -56, -50, -50), border = NA, col = "white")
    mtext(title3, adj = 0.1, line = 0.0)
    mtext(title2, side = 2, line = -1.5)
    if (add_legend) {
        add_raster_legend2(cols, limits, dat = trend[[2]],
                           transpose = FALSE, srt = 0, oneSideLabels= FALSE,
                           plot_loc = c(0.35, 0.99, 0.09, 0.12),  ylabposScling=0.8, ...)
    }
}