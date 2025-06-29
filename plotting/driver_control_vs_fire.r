library(raster)

dir0 = 'outputs/outputs_scratch/ConFLAME_nrt-drivers6/<<region>>-2425/samples/_21-frac_points_0.5/baseline-/'

region = 'Amazon'

dirs = c("Burned Area" = "Evaluate", "Fuel" = "Standard_0", "Moisture" = "Standard_1",
         "Weather" = "Standard_2")
        # "Wind" = "Standard_3", "Ignitions" = "Standard_4", 
        # "Suppression" = "Standard_5")


cols = c("#cfe9ff", "#fc6", "#f68373", "#c7384e", "#862976")


openDat <- function(dir, nfiles = 100, layers = 253:255) {
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
}

dat = lapply(dirs, openDat)

logit <- function(x) {
    x = (x+0.0000000001)/1.0000000002
    log(x/(1-x))
}

get_surface <- function(x, y, z) {
    # Create a grid
x.grid <- seq(min(x), max(x), length.out = 100)
y.grid <- seq(min(y), max(y), length.out = 100)

# Create an empty matrix for the surfaces
z10 <- matrix(NA, nrow = length(x.grid), ncol = length(y.grid))
z90 <- matrix(NA, nrow = length(x.grid), ncol = length(y.grid))

# Fill z10 and z90 with quantiles of z at each grid cell
for (i in 1:length(x.grid)) {
    print(i)
  for (j in 1:length(y.grid)) {
    # Find points near (x.grid[i], y.grid[j])
    d <- sqrt((x - x.grid[i])^2 + (y - y.grid[j])^2)
    nearby <- which(d < 0.5)  # radius threshold

    if (length(nearby) > 5) {
      z10[i, j] <- quantile(z[nearby], 0.1, na.rm = TRUE)
      z90[i, j] <- quantile(z[nearby], 0.9, na.rm = TRUE)
    }
  }
}

# Transpose to match x/y dimensions for plotting
z10 <- t(z10)
z90 <- t(z90)
return(x.grid, y.grid, z10, z90)
}

plot_pairs <- function(i, j) {
    if (i < j) {
        plot.new()
        return()
    }
    z = unlist(dat[[1]])
    limits = quantile(z, seq(0, 1, length.out = 6))
    z = cut_results(z, limits)
    x = logit(unlist(dat[[i+1]]))
    if (i == j)  {
        plot(x, z)
        return()
    }
    y = logit(unlist(dat[[j+1]]))
    #plot(x, y, type = 'n', axes = FALSE)
    #for (cex in c(2, 1, 0.5, 0.25, 0.1)) points(x, y, pch = 19, col = cols[z])
    surface = get_surface(x, y, z)
    browser()
    set_axes_pos <- function(x, axis) {
        xlab = c(0, 1, 10, 25, 50, 75, 90, 99, 100)
        pos = logit(xlab/100)
    
        if (sum((pos > min(x)) & (pos < max(x))) < 4) browser()
        axis(axis, at = pos, labels = xlab)
    }
    if (i == 1) set_axes_pos(y, 2)
    browser()
}
nplts = length(dirs) - 1
par(mfrow = c(nplts, nplts), mar = rep(1, 4), oma = rep(1.5, 4))
for (i in 1:nplts)
    for (j in 1:nplts)
        plot_pairs(i,j)        

