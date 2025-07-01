library(raster)
library(interp)
source("libs/plot_raster_map.r")

dir0 = 'outputs/outputs_scratch/ConFLAME_nrt-drivers6/<<region>>-2425/samples/_21-frac_points_0.5/baseline-/'

regions = c('Pantanal', 'Amazon', 'LA', 'Congo')

dirs = c("Burned Area" = "Evaluate", "Fuel" = "Standard_0", "Moisture" = "Standard_1",
         "Weather" = "Standard_2",
         "Wind" = "Standard_3", "Ignitions" = "Standard_4", 
         "Suppression" = "Standard_5")


cols = c("#cfe9ff", "#fc6", "#f68373", "#c7384e", "#862976")


openDat <- function(dir, region, nfiles = 100, layers = 253:255) {
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

logit <- function(x) {
    x = (x+0.000001)/1.000002
    log(x/(1-x))
}
logistic <- function(x) 
    return(1/(1+exp(-x)))

plot_BA_vs_control <- function(i, dat, y0, y) {
    x = logit(unlist(dat[[i]]))
    
    z = 100*(-y0 + y0/unlist(dat[[i]]))
    x = x[y>-12]
    z = z[y > -12]
    y = y[y >-12]
    
    cols = cols[unlist(mapply(rep, 1:5, 9 + (1:5)^3))]
    Dplot <- function(xi, yi) {
        cols = densCols(xi,yi, colramp = colorRampPalette(cols), bandwidth = 0.1) 
        plot(yi~xi, pch = 20, col = cols, cex = 1, axes = FALSE)
    }
    set_axes_pos <- function(x, axis,
                             xlab = c(0, 0.1, 0.2, 0.5, 1, 10, 25, 50, 
                                      75, 90, 99, 99.5, 99.8, 99.9, 100)) {
        
        pos = logit(xlab/100)
    
        #if (sum((pos > min(x)) & (pos < max(x))) < 4) 
            #set_axes_pos(x, axis,
        #                 c(0, unique(1-signif(1-logistic(seq(range(x)[1], range(x)[2], length.out = 10)), 1)), 1)*100)
        axis(axis, at = pos, labels = xlab)
    }
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
    Dplot(y, z)
    set_axes_pos(y, 1)
    if (i == 2) mtext(side = 3, 'BA vs Potential BA increase')
    if (i == length(dirs)) mtext(side = 1, 'BA', line = 2)
    axis(2)

}
plot_region <- function(region) {
    rm(map_mask, envir = .GlobalEnv)
    dat = lapply(dirs, openDat, region)
    rm(map_mask, envir = .GlobalEnv)
    y0 = unlist(dat[[1]])
    y = logit(y0)

    #plot_BA_vs_control(3)
    png(paste0("figs/nrt-control-scatter", region, ".png"), height = 12, width = 8, 
        units = 'in', res = 300)
    par(mfrow = c(6, 3), mar = rep(1, 4), oma = c(3, 3, 2, 2))
    sapply(2:7, plot_BA_vs_control, dat, y0, y)
    dev.off()
}
lapply(regions, plot_region)
browser()

get_surface <- function(x, y, z) {
    # Create a grid

# Define bin resolution
n_bins <- 100
x_breaks <- seq(min(x), max(x), length.out = n_bins + 1)
y_breaks <- seq(min(y), max(y), length.out = n_bins + 1)

# Bin x and y
x_bin <- cut(x, breaks = x_breaks, labels = FALSE, include.lowest = TRUE)
y_bin <- cut(y, breaks = y_breaks, labels = FALSE, include.lowest = TRUE)

# Combine into indices
bin_index <- (y_bin - 1) * n_bins + x_bin  # linear index

# Compute 10th and 90th percentile z per bin
z10_vals <- tapply(z, bin_index, function(v) quantile(v, 0.10, na.rm = TRUE))
z90_vals <- tapply(z, bin_index, function(v) quantile(v, 0.90, na.rm = TRUE))

# Create empty matrices for plotting
z10_mat <- matrix(NA, nrow = n_bins, ncol = n_bins)
z90_mat <- matrix(NA, nrow = n_bins, ncol = n_bins)

# Fill matrices
z10_mat[as.numeric(names(z10_vals))] <- z10_vals
z90_mat[as.numeric(names(z90_vals))] <- z90_vals

# Define grid centers for plotting
x_mid <- 0.5 * (x_breaks[-1] + x_breaks[-length(x_breaks)])
y_mid <- 0.5 * (y_breaks[-1] + y_breaks[-length(y_breaks)])







valid <- which(!is.na(z10_mat), arr.ind = TRUE)

x_valid <- x_mid[valid[, 2]]
y_valid <- y_mid[valid[, 1]]
z10_vals <- z10_mat[valid]
z90_vals <- z90_mat[valid]

# Interpolate smooth quantile surfaces
interp_10 <- interp(x_valid, y_valid, z10_vals, linear = FALSE, extrap = TRUE)
interp_90 <- interp(x_valid, y_valid, z90_vals, linear = FALSE, extrap = TRUE)


loc = list(seq(x_valid[1], tail(x_valid, 1),,500), seq(y_valid[1], tail(y_valid, 1),,500))
loc = make.surface.grid(loc)
interp_10 = as.surface(loc, interp.surface( interp_10, loc))
interp_90 = as.surface(loc, interp.surface( interp_10, loc))
      
return(list(interp_10, interp_90))

}

library(ggplot2)
library(gridExtra)
    

plot_pairs <- function(i, j) {
    if (i > j) {
        plot.new()
        return()
    }
    
    z = (unlist(dat[[1]]))
    limits = quantile(z, seq(0, 1, length.out = 6))
    x = logit(unlist(dat[[i+1]]))

    
    if (i == j)  {
        z = logit(z)
        plot(x, z, axes = FALSE)
        set_axes_pos(x, 1)
        set_axes_pos(z, 2)
        
        return()
    }
    y = logit(unlist(dat[[j+1]]))
    
    z = cut_results(z, limits)
    plot(x, y, type = 'n', axes = FALSE)
    for (cex in c(2, 1, 0.5)) points(x, y, pch = 19, col = cols[z])
    if (F) {
    surface = get_surface(x, y, z)
    if ( i < j) {
        z_n = 1
    } else {
        z_n = 2
    }
   
    image.plot( surface[[z_n]], col= colorRampPalette(cols)(20),add = TRUE)
    }
    
    
    if (i == 1) set_axes_pos(x, 3)
    if (i == nplts) set_axes_pos(x, 1)

    if (j == 1) set_axes_pos(x, 2)
    if (j == nplts) set_axes_pos(x, 4)
    
}
nplts = length(dirs) - 1
par(mfrow = c(nplts, nplts), mar = rep(1, 4), oma = rep(1.5, 4))
for (i in 1:nplts)
    for (j in 1:nplts)
        plot_pairs(i,j)        

