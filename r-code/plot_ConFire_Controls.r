library(raster)
source("../gitProjectExtras/gitBasedProjects/R/sourceAllLibs.r")
sourceAllLibs("../rasterextrafuns/rasterPlotFunctions/R/")
sourceAllLibs("../rasterextrafuns/rasterExtras/R/")
sourceAllLibs("../gitProjectExtras/gitBasedProjects/R/")
library(ncdf4)
sourceAllLibs("../ConFIRE_attribute/libs/")
source("../ConFIRE_attribute/libs/plotStandardMap.r")
source("../LPX_equil/libs/legendColBar.r")
source("libs/find_levels.r")
graphics.off()

colss = list(c('#ffffe5','#f7fcb9','#d9f0a3','#addd8e','#78c679','#41ab5d','#238443','#006837','#004529'), 
            c('#ffffd9','#edf8b1','#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8','#253494','#081d58'),
            c('#fff7f3','#fde0dd','#fcc5c0','#fa9fb5','#f768a1','#dd3497','#ae017e','#7a0177','#49006a'),
            c('#ffffff','#f0f0f0','#d9d9d9','#bdbdbd','#969696','#737373','#525252','#252525','#000000'))

dir = 'outputs/ConFire_UK/samples/crop_lightn_soilM_trees_csoil_pas_vpd_cveg_precip_tas_rhumid_totalVeg-frac_points_0.5/Standard_'

controls = c('Fuel', 'Moisture', 'Ignitions', 'Suppression')

plot_control <- function(i, name, cols) {
    files = list.files(paste0(dir, i-1), full.names = TRUE)
    dats = layer.apply(files[1:20], function(i) mean(brick(i)))
    dats[dats>9E9] = NaN
    mn = mx = dats[[1]]
    qu = apply(dats[], 1, quantile, c(0.25, 0.75), na.rm = TRUE)
    mn[] = qu[1,]
    mx[] = qu[2,]
    
    levels = find_levels(c(mn[!is.na(mn)], mx[!is.na(mx)]), seq(10, 90, 10))
    plotStandardMap(mn, cols = cols, limits = levels)
    if (i == 1) mtext(side = 3, '10%', adj = 0.15, line = -2)
    plotStandardMap(mx, cols = cols, limits = levels)
    if (i == 1) mtext(side = 3, '90%', adj = 0.85, line = -2)
    mtext(name, adj = -0.4, xpd = TRUE, line = -1.5)
    legendColBar(c(0.1, 0.7), c(0.1, 0.9), cols = cols, limits = levels, extend_min = F, minLab = 0)
}
png("r-code/UK_ConFire_histMaps.png", height = 4, width = 7, res = 300, units = 'in')
layout(rbind(c(1:3, 7:9), c(4:6, 10:12)), widths = c(1, 1, 0.5, 1, 1, 0.5))
par(mar = c(0, 1, 0, 0))
mapply(plot_control, 1:length(controls), controls, colss)
dev.off()