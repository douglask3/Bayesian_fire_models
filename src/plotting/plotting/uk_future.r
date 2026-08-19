

dir = "outputs/outputs_scratch/UK/isimip/test-ukSPECIFIC5/time_series/_14-frac_points_1e-24/"

base = "historical"
ssps = c("ssp126", "ssp370", "ssp585")

models = c("IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL")#, "GFDL-ESM4")

years = seq(2000, 2090, by = 10)
targeMnths = c('06', '07', '08')

annual_average <- function(dat, fun = mean) {

    xindex = apply(dat[,1:240],1 ,mean)>0.001 & apply(dat[,1:240],1 ,mean)<0.1
    decade <- function(year) {
        selectYr <- function(yr) which(substr(colnames(dat), 2, 5) == yr)
        #selectMn <- function(mn) substr(colnames(dat), 7, 8) == mn
        index = unlist(lapply(year:(year+9), selectYr))  
        #index2 = unlist(lapply(targeMnths, selectMn)) 
        
        return(apply(dat[,index], 1, mean))
    }   
    out = lapply(years,decade)
    return(do.call(cbind,out))
    browser()
}

open_period <- function(dir) {
    filename = paste0(dir, '/mean/members/absolute/Control.csv')
    return(read.csv(filename, stringsAsFactors=F)[,-1])
}

open_model <- function(model, ssp, fun = annual_average) {

    period = list.dirs(paste0(dir, base, '/', model), recursive = FALSE)
    dat0 = open_period(period)
    periods = list.dirs(paste0(dir, ssp, '/', model), recursive = FALSE)
    
    
    dat = lapply(periods, open_period)
    dats = cbind(dat0, do.call(cbind, dat))
    return(fun(dats))
}

open_ssp <- function(ssp) {
    outs = lapply(models, open_model, ssp)
    return(do.call(rbind,outs)*24437.6)
}


dats = lapply(ssps, open_ssp)

normalise_dat <- function(dat) apply(dat, 2, function(i) (i)/(dat[,1]))

dats = lapply(dats, normalise_dat)
yrange = range(sapply(dats, quantile, c(0.01, 0.985), na.rm = TRUE))
layout(rbind(c(2, 1), c(1, 1)),widths = c(0.6, 0.8), heights = c(0.4, 0.8))

plot(range(years) + c(5, 5), yrange, xlab = '', ylab = '', type = 'n', xaxt = 'n')
axis(1, at = seq(2015,2095, 10), labels = paste(seq(2010,2090, 10), 's'))

add_ssp <- function(dat, name, col, offset, width = 1) {
    for_year <- function(i, year) {
        x = year + offset
        ys = quantile(dat[,i], c(0.05, 0.25, 0.5, 0.75, 0.95), na.rm = TRUE)
        for_line <- function(j, wd = width, ...) {
            lines(x + c(-wd, wd),c(ys[j], ys[j]),  ...)
        }
        lines(c(x, x), ys[c(1, 5)])
        polygon(x + c(-1, -1, 1, 1)*width*2/3, ys[c(2, 4, 4, 2)], col = col, border = NA)
        
        for_line(1)
        for_line(5)
        for_line(3, wd = width*2/3, lwd = 2)
        lines(c(x, x), ys[c(1, 5)], col = '#00000099')
        text(x = x, y = ys[1], name, adj = c(0.5, 1.5), xpd = NA)
    }
    
    mapply(for_year, 2:length(years), years[-1])
    
}
lapply(seq(2000, 2100, by = 10), function(x) lines(c(x, x), c(-9E9, 9E9), col = 'black', lty = 3))
lapply(seq(0, 10, by = 0.1), function(y) lines( c(-9E9, 9E9), c(y, y),  col = 'grey', lty = 3))
lines( c(-9E9, 9E9), c(1, 1),  col = '#333333', lty = 1)


add_ssp(dats[[1]], '', '#008787', 3)
add_ssp(dats[[2]], '', '#E27226', 5)
add_ssp(dats[[3]], '', '#C7403D', 7)



plot(c(10, 20), c(0, 1), xlab = '', ylab = '', type = 'n', axes = FALSE)
years = c(0, 10)
mat = matrix(runif(2000, 0.1, 0.99), ncol = 2)
polygon(c(10, 10, 20, 20), c(-0.2, 1, 1, -0.2), col = 'white', border = NA, xpd = NA)
add_ssp(mat, ssps[1], '#008787', 2.5)
add_ssp(mat, ssps[2], '#E27226', 5)
add_ssp(mat, ssps[3], '#C7403D', 7.5)

