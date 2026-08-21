

dir = "outputs/outputs_scratch/UK/isimip/test-ukSPECIFIC9-UK/time_series/_14-frac_points_1e-24/"

base = "historical"
ssps = c("ssp126", "ssp370", "ssp585")

cols = c('#008787', '#E27226', '#C7403D')

models = c("IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL")#, "GFDL-ESM4")

years = seq(2000, 2090, by = 10)
targeMnths = c('06', '07', '08')

annual_average <- function(dat, datF, datM, datH, fun = mean) {

    #xindex = apply(dat[,1:240],1 ,mean)>0.001 & apply(dat[,1:240],1 ,mean)<0.1
    datF = apply(datF, 1 ,mean)
    datM = apply(datM, 1 ,mean)
    
    dat0 = apply(datH[,121:240], 1, mean)
    prob = ((1-exp(-1+datM))*(1-exp(-1+datF)))^(1/2)
    BA = 0.00017
    prob = (exp(BA*log(dat0) + (1.0-BA)*log((1-dat0)))) * prob
    #prob[dat0>0.1] = 0
    set.seed(123458)
    
    samples = sample(1:nrow(dat), 200, TRUE, prob)
    decade <- function(year) {
        selectYr <- function(yr) which(substr(colnames(dat), 2, 5) == yr)
        index = unlist(lapply(year:(year+9), selectYr))  
        #browser()
        return(apply(dat[samples,index], 1, fun))
    }   
    out = lapply(years,decade)
    return(do.call(cbind,out))
}

open_period <- function(dir, filename = "Evaluate.csv") {
    filename = paste0(dir, '/mean/members/absolute/', filename)
    out = read.csv(filename, stringsAsFactors=F)[,-1]
               
}

open_model <- function(model, ssp, fun = annual_average, ...) {

    period = list.dirs(paste0(dir, base, '/', model), recursive = FALSE)
    dat0 = open_period(period, ...)
    datF = open_period(period, filename = "standard-Fuel.csv")
    datM = open_period(period, filename = "standard-Moisture.csv")
    datH = open_period(period, filename = "Evaluate.csv")
    periods = list.dirs(paste0(dir, ssp, '/', model), recursive = FALSE)
    
    
    dat = lapply(periods, open_period, ...)
    dats = cbind(dat0, do.call(cbind, dat))
    return(fun(dats, datF, datM, datH))
}

open_ssp <- function(ssp, ...) {
    outs = lapply(models, open_model, ssp, ...)
    return(do.call(rbind,outs)*24437.6)
}

af_tscale <- function(x) {
    y = x
    y[x>=1] = 1-1/x[x>=1]
    y[x<1] = -(1-x[x<1])
    return((y+1)/2)
}


mit_tscale <- function(x) {
    return(af_tscale((x/100) +1))
    x = x /100
    y= x
    y[x>=0]= 1-0.5/(x[x>=0]+1)
    y[x<0]= 1-(1-0.5/(-x[x<0]+1))
    return(y)
}

add_ssp <- function(dat, name, col, offset, 
                    width = 1, tplot = TRUE, transform = af_tscale, lab_bottom = T) {
    if (all(is.na(dat))) return(NULL)
    for_year <- function(i, year) {
        x = year + offset
        ys = quantile(dat[,i], c(0.05, 0.25, 0.5, 0.75, 0.95), na.rm = TRUE)
        if (!is.null(transform)) ys = transform(ys)
        if (!tplot) return(ys)
        for_line <- function(j, wd = width, ...) {
            lines(x + c(-wd, wd),c(ys[j], ys[j]),  ...)
        }
        #lines(c(x, x), range(dat[,i]), col = col)
        lines(c(x, x), ys[c(1, 5)])
        polygon(x + c(-1, -1, 1, 1)*width*2/3, ys[c(2, 4, 4, 2)], col = col, border = NA)
        
        for_line(1)
        for_line(5)
        for_line(3, wd = width*2/3, lwd = 2)
        lines(c(x, x), ys[c(1, 5)], col = '#00000099')
        if (lab_bottom)
            text(x = x, y = ys[1], name, adj = c(0.5, 1.5), xpd = NA)
        else
            text(x = x, y = tail(ys, 1), name, adj = c(0.5, -0.5), xpd = NA)
    }
    
    mapply(for_year, 2:length(years), years[-1])
    
}

add_legend <- function() {
    plot(c(10, 100), c(0, 1), xlab = '', ylab = '', type = 'n', axes = FALSE)
    
    mat = matrix(runif(2000, 0.1, 0.99), ncol = 2)
    #polygon(c(10, 10, 20, 20), c(-0.2, 1, 1, -0.2), col = 'white', border = NA, xpd = NA)
    add_ssp(mat, ssps[1], '#008787', 2.5, transform = NULL)
    add_ssp(mat, ssps[2], '#E27226', 5, transform = NULL, lab_bottom = F)
    add_ssp(mat, ssps[3], '#C7403D', 7.5, transform = NULL)
}

af_axis <- function() {
    labels = af_tscale(c(0, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 2/3, 1, 
               1.5, 2, 4, 8, 16, 32, 64, 128, 1000000))
    labels_txt = c('0', '', '', '', '', '1/8', '1/4', '1/2', '2/3', 'no\nchange', '3/2',
                   '2', '4', '8', '', '', '', '', 'All from\nclimate')
    ylim=par("usr")[3:4]
    if (sum(((labels) > ylim[1]) & ((labels) < ylim[2]))<5) {
        labels = af_tscale(c(0, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 2/3, 4/5, 1, 1.25,
               1.5, 2, 4, 8, 16, 32, 64, 128, 1000000))
        labels_txt = c('0', '', '', '', '', '', '1/4', '1/2', '2/3', '4/5', 
                       'no\nchange', '5/4', '3/2',
                       '2', '4', '', '', '', '', '', 'All from\nclimate')
    }
    axis(2, at = labels, labels = labels_txt)
    lapply(labels, function(y) lines( c(-9E9, 9E9), c(y, y),  col = 'grey', lty = 3))
    #browser()
}

mit_axis <- function() {
    labels = c(-100, -50, -25, -10, -5, 0, 5, 10, 25, 50, 100)
    ylim=par("usr")[3:4]
    if ( sum((mit_tscale(labels) > ylim[1]) & (mit_tscale(labels) < ylim[2]))<5) 
        labels = labels/2
    at = mit_tscale(labels)
    axis(2, at = at, labels = labels)
    lapply(at, function(y) lines( c(-9E9, 9E9), c(y, y),  col = 'grey', lty = 3))
}

af_plot <- function(dats, pname, fun = af_tscale, axis_fun = af_axis) {
    
    yrange = mapply(add_ssp, dats, '', cols, c(3, 5, 7), 
                    MoreArgs = list(tplot = FALSE, transform = fun))

    yrange = as.vector(yrange)
    
    yrange = range(yrange)
    if (yrange[1] < 0) yrange[1] = 0
    yrange = yrange + c(-1, 1) * diff(yrange) * 0.04

    if (yrange[1] < 0) yrange[1] = 0
    if (yrange[2] > 1) yrange[2] = 1
    plot(range(years) + c(10, 5), yrange, xlab = '', ylab = '', type = 'n', xaxt = 'n', yaxt = 'n', yaxs = 'i')
    axis_fun()

    
    mtext(side = 3, adj = 0.1, pname, font = 2)
    axis(1, at = seq(2015,2095, 10), labels = paste(seq(2010,2090, 10), 's'))
    

    lapply(seq(2000, 2100, by = 10),          
           function(x) lines(c(x, x), c(-9E9, 9E9), col = 'black', lty = 3))
    
    lines( c(-9E9, 9E9), c(1, 1),  col = '#333333', lty = 1)
    
    mapply(add_ssp, dats, '', cols, c(3, 5, 7), MoreArgs = list(transform = fun))
}

mit_plot <- function(dats) {
    dats[[1]] = -100*((dats[[2]]/dats[[1]])-1)
    dats[[2]] = -100*((dats[[3]]/dats[[2]])-1)
    dats[[3]][] = NaN
    af_plot(dats, rep('', 3), fun = mit_tscale, axis_fun = mit_axis)
}


plot_output <- function(filename = "Evaluate.csv", pname = "Burned Area") {
    dats = lapply(ssps, open_ssp, filename = filename)
    normalise_dat <- function(dat) apply(dat, 2, function(i) (i)/(dat[,1]))
    dats = lapply(dats, normalise_dat)
    af_plot(dats, pname) 
    mit_plot(dats)
}
graphics.off()
png("figs/UK_proj_ts.png", height = 10, width = 7.2, res = 300, units = 'in')

    layout(rbind(1:2, 3:4, 5:6, c(7,0)), heights = c(1,1, 1, 0.5))
    par(mar = c(2.5, 3, 1.5, 1))
    plot_output()
    plot_output("standard-Moisture.csv", "Dryness")
    plot_output("standard-Fuel.csv", "Fuel")
    #par(mar = c(0, 1, 1,0))
    years = c(0, 10)
    add_legend()
graphics.off()
