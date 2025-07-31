graphics.off()
library(raster)
source("libs/find_levels.r")
source("../rasterextrafuns/rasterPlotFunctions/R/make_col_vector.r")
source("../rasterextrafuns/rasterPlotFunctions/R/plot_raster_map.r")
source("../rasterextrafuns/rasterExtras/R/layer.apply.r")
source("../rasterextrafuns/rasterExtras/R/is.raster.class.r")
source("../rasterextrafuns/rasterExtras/R/addLayer.ext.r")
source("../LPX_equil/libs/legendColBar.r")
source("../rasterextrafuns/rasterPlotFunctions/R/mtext.units.r")
region = 'NW_Amazon'

region.name = c('Canada' = 'Canada', 'Greece' = 'Greece', 'NW_Amazon' = 'Western Amazonia')

BA_obs_file = paste0('data/data/driving_data2425/', region, '/nrt/era5_monthly//burnt_area.nc')

BA_mod_dir_alls = paste0('outputs/ConFire_', region, '-nrt-tuning10/samples/_12-frac_points_0.5/baseline-/', c('Evaluate', 'control'), '/')

mn = 9:10

cols = c('#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026')
dcols = rev(c('#67001f','#b2182b','#d6604d','#f4a582','#fddbc7','#f7f7f7','#d1e5f0','#92c5de','#4393c3','#2166ac','#053061'))

openFile <- function(file) {
    out = brick(file)
    dates = names(out)
    clim_index = grepl(paste0('.0', mn, '.'), dates, fixed = TRUE) & substr(dates, 2, 5) > 2013
    clim = mean(out[[which(clim_index)]])*100
    
    index = grepl('2023', dates) & clim_index
    sow = out[[which(index)]]*100
    anom = sow/clim
    anom[anom<1] = 2-1/anom[anom<1]
    anom = anom - 1
    out = addLayer(clim, sow, anom)
    out[out[[1]] > 9E9] = NaN
    return(out)
}

BA_obs = openFile(BA_obs_file)


insert_zeros <- function(mat) {
  # Get the dimensions of the original matrix
  nrow_original <- nrow(mat)
  ncol_original <- ncol(mat)
  
  # Calculate the dimensions of the new matrix
  nrow_new <- 2 * nrow_original + 1
  ncol_new <- 2 * ncol_original + 1
  
  # Create a new matrix filled with zeros
  new_mat <- matrix(0, nrow = nrow_new, ncol = ncol_new)
  
  # Insert the original matrix elements into the new matrix
  new_mat[seq(2, nrow_new, by = 2), seq(2, ncol_new, by = 2)] <- mat
  
  return(new_mat)
}

find_cex <- function(r) {
    # Get the plot dimensions in user coordinates
    #usr <- par("usr") # returns c(x1, x2, y1, y2)
    #plot_width_user <- usr[2] - usr[1] # width in user coordinates

    # Get the plot dimensions in inches
    pin <- par("pin") # returns c(width, height) in inches
    plot_width_inch <- pin[1] # width in inches
    return(plot_width_inch*r*13.2)
    # Calculate the desired point size in user coordinates
    point_size_user <- plot_width_user * r
    
    # Convert the point size to 'cex'
    # By default, the 'cex' value of 1 corresponds to a default point size of 0.1 inches
    default_point_size_inch <- 0.1

    # Calculate 'cex' value
    cex <- point_size_user / (plot_width_inch * default_point_size_inch)
}

plotCells <- function(xyzs, res, cols, border = NA) {
    addCell <- function(xyz) {
        polygon(xyz[1] + res[1] * c(-1, 1, 1, -1, -1), xyz[2] + res[2] * c(-1, -1, 1, 1, -1), 
                col = cols[xyz[3]], border = border)
    }
    apply(xyzs, 1, addCell)
}

plotPoints <- function(xyzs, res, cols) {

    cexs = sapply(1:length(cols), function(j) apply(xyzs[,-(1:2)], 1, 
                                    function(x) mean(x>=j, na.rm = TRUE)))^0.5
    
    if (diff(range(xyzs[,1]))/res[1] > 50) {
        y = seq(min(xyzs[,2]), max(xyzs[,2]), length.out = 30)
        x = seq(min(xyzs[,1]), max(xyzs[,1]), by = diff(y[1:2]))
        xydat_new = cbind(rep(x, length(y)), rep(y, each = length(x)))

        assign_xy <- function(xy) 
            which.min((xy[1] - xydat_new[,1])**2 + (xy[2] - xydat_new[,2])**2)
       
        id = apply(xyzs, 1, assign_xy)
        
        vcdat_new = matrix(NaN, ncol = dim(cexs)[2], nrow = nrow(xydat_new))
        for (i in 1:nrow(xydat_new)) {
        
            test = which(id == i)
            #if (length(test) > 0 )browser()
            if (length(test) == 1) vcdat_new[i,] = cexs[test,]
            else if (length(test) > 1 ) vcdat_new[i,] = apply(cexs[test,], 2, mean)
        }
        test = !is.na(vcdat_new[,1])
        
        xyzs = xydat_new[test,]
        cexs = vcdat_new[test,]
        res = apply(diff(xydat_new), 2, function(x) min(x[x>0]))
    }
    
    cex_scale = find_cex(res[1]/diff(range(xyzs[,1])))
    cexs = cex_scale * cexs
    points(xyzs[,1], xyzs[,2], cex = cex_scale * 1.01, pch = 19)
    points_level <- function(i) 
        points(xyzs[,1], xyzs[,2], col = cols[i], cex = cexs[,i], pch = 19)

    lapply(1:length(cols), points_level)
}

plot_raster_image <- function(r, cols, levels = NULL, cntr = NULL, ..., plotFun = plotCells) {
    
    if (is.null(levels))  {
        levels =  find_levels_n(r, 6, TRUE)  
        levels = unique(sort(c(levels/10, levels)))
    }
    cols = make_col_vector(cols, ncols = length(levels) + 1)
    
    r = cut_results(r, levels)
    mask = any(!is.na(r))
    if (is.null(cntr)) cntr = mask
    
    xyzs = cbind(xyFromCell(r, (1:length(r[[1]][]))[mask[]]), r[mask[]])
    res = res(r)
    
    plot(extent(r)[1:2], extent(r)[3:4], type = 'n', axes = FALSE, xlab = '', ylab = '')
    grid()
    plotFun(xyzs, res, cols, ...)
    contour(cntr, levels = 0.01, drawlabels=FALSE, add = T, lwd = 2, xpd = NA)
    return(list(levels, mask))
}

plot_raster_points <- function(...) 
    plot_raster_image(..., plotFun = plotPoints)

#par(mfcol  = c (3, 2))
lmat = rbind(c( 2,  0,  1,  1,  0,  3), 0,
             c( 4,  0,  5,  5,  0,  6), 0,
             c( 7,  0,  8,  8,  0,  9), 
             c(10, 10, 10, 11, 11, 11))
lmat = cbind(0, rbind(0, lmat, 0), 0)

pltHght = nrow(BA_obs)/ncol(BA_obs)
heights = c(0.5, pltHght, 0.05, pltHght, 0.05, pltHght, 0.4, 0.05)
widths = c(0.35, 1, 0.05, 0.5, 0.5, 0.05, 1, 0.2)
png(paste0("figs/ba_anaom_nrt-2-", region, ".png"), res = 300, height = sum(heights)*3,
    width = sum(widths)*3, units = 'in')
layout(lmat, width = widths, heights = heights)
par(mar = rep(0, 4))

#BA_obs[is.na(BA_mod_clim)] = NaN
levels = plot_raster_image(BA_obs[[2]], cols)
cntr = levels[[2]]; levels = levels[[1]]
axis(3)
mtext('2023 BA %', line = 2)
mtext(line = 3.5, paste0(region.name[region], ' - ',paste0( month.name[mn], collapse = ', ')), 
      font = 2)
plot_raster_image(BA_obs[[1]], cols, levels, cntr = cntr)
axis(2); axis(3)
mtext('Observations', side = 2, line = 2)
mtext('Mean 2014-2023 BA %', line = 2)
dlevels = plot_raster_image(BA_obs[[3]]-1, dcols, cntr = cntr)[[1]]
axis(3); axis(4)
mtext('2023 BA: 2014-2023 BA', line = 2)

plot_dir <- function(BA_mod_dir_all, name) {
    BA_mod_files = list.files(BA_mod_dir_all, full.names = TRUE)
    BA_mod_files = BA_mod_files[grepl('-pred', BA_mod_files)]
    BA_mod = lapply(BA_mod_files, openFile)
    BA_mod_clim = layer.apply(BA_mod, function(r) r[[1]])
    BA_mod_sow  = layer.apply(BA_mod, function(r) r[[2]])
    BA_mod_anom = layer.apply(BA_mod, function(r) r[[3]])
    plot_raster_points(BA_mod_clim, cols, levels, cntr = cntr)
    axis(2)
    mtext(name, side = 2, line = 2)
    if (BA_mod_dir_all == tail(BA_mod_dir_alls, 1)) axis(1)
    plot_raster_points(BA_mod_sow, cols, levels, cntr = cntr)
    if (BA_mod_dir_all == tail(BA_mod_dir_alls, 1)) axis(1)
    plot_raster_points(BA_mod_anom, dcols, levels, cntr = cntr)
    axis(4)
    if (BA_mod_dir_all == tail(BA_mod_dir_alls, 1)) axis(1)
}
mapply(plot_dir, BA_mod_dir_alls, c('With\nStocasticity', 'Just drivers'))
legendColBar(xx = c(0.1, 0.4), yy = c(0.0, 1.0), cols = cols, limits = levels, 
             transpose = TRUE, oneSideLabels=FALSE, extend_min = F, minLab = 0, units = '%')

legendColBar(xx = c(0.1, 0.4), yy = c(0.0, 1.0), cols = dcols, limits = dlevels, 
             transpose = TRUE, oneSideLabels=FALSE)
dev.off()
