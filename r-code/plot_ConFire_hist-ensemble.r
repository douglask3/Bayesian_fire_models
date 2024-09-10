graphics.off()
library(terra)

##########
## cfg ###
##########
region = 'Greece'
date_to_test = '2023-08'

fname = paste0("outputs/ConFire_", region, "-tuning3/figs/_13-frac_points_0.1-")
fileEx = "-control_TS/points-Control.csv"

burnt_area_data = paste0("data/data/driving_data/", region, "/isimp3a/obsclim/GSWP3-W5E5/period_2000_2019/burnt_area-2000-2023.nc")

trans_fun <- function(x) log10(x)
itrans_fun <- function(x) 10^x

trans_fun <- function(x) x*100
itrans_fun <- function(x) x


##############################
## cal. burnt area metrics ###
##############################
rast_climatelogy_month <- function(r) {
    forMonth <- function(mn) {
        index = seq(mn, nlyr(r), by = 12)
        r[[index]] = r[[index]] - mean(r[[index]])
        return(r)
    }
    for (mn in 1:12) r = forMonth(mn)
    return(r)
}

vector_climatelogy_month <- function(x, x0 = x) {
    return(x)
    forMonth <- function(mn) {
        index = seq(mn, length(x), by = 12)
        x[index] = x[index] - mean(x0[index])
        return(x)
    }
    
    for (mn in 1:12) x = forMonth(mn)
    return(x)
}


burnt_area = rast(burnt_area_data)
#burnt_area = rast_climatelogy_month(burnt_area)


date_test = substr(time(burnt_area), 1, 7) == date_to_test
month_to_test = as.numeric(substr(date_to_test, nchar(date_to_test)-1, nchar(date_to_test)))
#burnt_area_event = burnt_area[[date_test]]
gridArea = cellSize(burnt_area)

burnt_area_tot = sapply(1:nlyr(burnt_area), function(i) sum((burnt_area[[i]] * gridArea)[], na.rm = TRUE)) / sum(gridArea[], na.rm = T)

burnt_area_event = burnt_area_tot[date_test]

percentile = mean(burnt_area_tot <= burnt_area_event)

extreme = trans_fun(burnt_area_event)
plot_experiments <- function(experiments, conPeriod = "today", expPeriod = "by 2090",
                            title = '', histID = 2) {
    cols = c('#0000FF', '#FF0000')
    openDat <- function(exp, ny) {
        dat = tail(t(trans_fun(0.000001+read.csv(paste0(fname, exp, fileEx)))), ny*12)
    }
    
    dats = lapply(experiments, openDat, ny = 19)
    if (length(dats)>2) browser()#dats = c(dats[1], list(unlist(dats[-1])))
    
    open_for_realization <- function(i) {
        dat = sapply(dats, function(dt) vector_climatelogy_month(dt[,i], dats[[histID]][,i]))#
        
        #vector_climatelogy_month(dt[,i]))   
        scale = extreme/quantile(dat[,histID], c(percentile))
        
        dat = dat * scale
        dat = dat[seq(2, nrow(dat), by = 3),]
        browser()
        return(dat)
    }
    
    dats = lapply(1:(dim(dats[[1]])[2]), open_for_realization)
    bins = range(unlist(dats)); bins = seq(bins[1], bins[2], 
                     length.out = floor(sqrt(length(unlist(dats[[1]])))))

    process_for_realization <- function(dat) {
        pr = apply(dat, 2, function(x) mean(x>extreme))
        pr = pr[2]/pr[1]

        mean_change = mean(dat[,2]) - mean(dat[,1])

        ys = apply(dat, 2, function(x) hist(unlist(x), bins, plot = FALSE)$density)
        #ys = lapply(ys, function(i) log(i + 1))
        return(list(ys, pr, mean_change))
    }
    binned = sapply(dats, process_for_realization)
    browser()
    x = bins[-1] - diff(bins)
    
    plot(range(x, extreme), range(unlist(ys)), 
         type = 'n', axes = FALSE, xlab = '', ylab = '')
    #axis(1, at = seq(-100, 100), itrans_fun(seq(-100, 100)))
    axis(1)
    mtext(title, line = -1, side = 3, font = 2)
    addPoly <- function(y, col) 
        polygon(c(x[1], x, tail(x, 1)), c(log(1), y, log(1)), col = paste0(col, '44'), 
                border = NA)

    mapply(addPoly, c(ys, rev(ys)), c(cols, rev(cols)))
    
    #extreme = burnt_area#quantile(unlist(dats[[1]]), 0.99)
    lines(rep(extreme, 2), c(0, 9E9), lty = 2)
    
    extreme_fut = round(mean(dats[[2]] > extreme)/mean(dats[[1]] > extreme), 3)
    #mtext(side = 3, adj = 0.5, line = -4, paste0("1-in-100 ", conPeriod,  " occurs\n1-in-", 
    #                                             extreme_fut, " ", expPeriod))
    
    mtext(side = 3, adj = 0.5, line = -4, paste0("2023 risk ratio: ", extreme_fut))

}
png(paste0(region, "_attribution.png"), height = 3, width = 7, units = 'in', res = 300)
par(mfrow = c(1, 3), mar = c(3, 0.5, 3, 0.5))
plot_experiments(c("counterfactual", "factual"), "at PI", "today")
#plot_experiments(c("factual", "ss126_GFDL", "ss126_IPSL", "ss126_MPI", "ss126_MRI"), title = "ssp126")
#plot_experiments(c("factual", "ss585_GFDL", "ss585_IPSL", "ss585_MPI", "ss585_MRI"), title = "ssp585")
dev.off()               
