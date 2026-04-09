graphics.off()
library(terra)

cols = c("#cfe9ff", "#fc6", "#f68373", "#c7384e", "#862976")
cols = c("#7a44ff", "#0096a1", "#e98400", "#b50000")
target_year = 2024
dir = 'data/data/driving_data_base/'


open_BA_dat <- function(region, q95) {
    dir = paste0(dir, region, '/nrt/era5_monthly/')
    ba_file = list.files(dir, full.names=TRUE)
    ba_file = ba_file[grepl('Fire_f', ba_file)][2]
    
    dat = rast(ba_file)
    
    #if (q95) {
    #    for (i in 1:nlyr(dat)) {
    #        print(i)
    #        q = quantile(dat[[i]][], 0.95, na.rm = T)
    #        dat[[i]][dat[[i]]<q] = NaN
    #    }
    #}
    
    area = cellSize(dat[[1]])
    area0 = area
    area[is.na(dat)]   = NaN
    tArea = sum(area[], na.rm = TRUE)
    BA_4_layer <- function(i) {
        print(i)
        if (q95) {
            q = quantile(dat[[i]][], 0.95, na.rm = T)
            dat[[i]][dat[[i]]<q] = NaN
            area = area0
            area[is.na(dat[[i]])]   = NaN
            tArea = sum(area[], na.rm = TRUE)
        }
        sum(values(dat[[i]]*area), na.rm = TRUE)/tArea
    }
    BA = sapply(1:nlyr(dat), BA_4_layer)
    
    time = time(dat)
    cBA = sapply(1:12, function(mn) mean(BA[seq(mn, length(BA), by = 12)]))
    cBA = rep(cBA, length.out = length(BA))
    
    return(list(time, BA, cBA))
}

plot_BA_type <- function(region, q95 = FALSE, xtitle = '', ytitle = '') {
    
    BA = open_BA_dat(region, q95)
    cBA = BA[[3]]
    time = BA[[1]]
    BA = BA[[2]]
    #time = dat$time
    
    #BA = dat[,name]/4
    #cBA = dat[,cname]/4  
    #BA = dBA
    y_range = range(BA)
    y_max = quantile(BA, 0.999 )/1.6
    #plot(c(0.5, 12.5), y_range, type = 'n', xlab = '', ylab = '', xaxt = 'n', yaxt = 'n')
    
    plot(0, 0, xlim = c(y_max * c(-1, 1)), ylim = c(y_max * c(-1, 1)), type = 'n', xlab = '', ylab = '', xaxt = 'n', 
         yaxt = 'n', asp = 1, axes = FALSE)#, xaxs = 'i', yaxs = 'i')

    mtext(side = 2, xtitle, adj = 0.83, font = 2, line = 0.75)
    mtext(side = 3, ytitle, adj = 1-0.83, font = 2, line = 0.75)
    #axis(2)
    #axis(1, at = 1:12, labels = month.abb)  
    #grid()
    #axis(1)
    #axis(2)
    at = par("xaxp")
    at = seq(at[1], at[2], length.out = at[3] + 1)
    #labels = abs(at)
    at = at[at>=0]
    
    axis(1, pos = 0, at = at, labels = at)
    axis(2, pos = 0, at = at, labels = at)
    
    axis(3, pos = 0, at = -at, labels = at)
    axis(4, pos = 0, at = -at, labels = at)


    at = c(at,y_max*1.1) 
    xgrid = seq(0, 12, 0.01)
    tgrid = - 2 * pi * (xgrid -3)/ 12
    add_grid_line <- function(y) {
        x_cart = y * cos(tgrid)
        y_cart = y * sin(tgrid)
        bigID = 1+(y == max(at))
        lines(x_cart, y_cart, lty = c(3, 1)[bigID], col = c('grey', 'black')[bigID], xpd = NA)
    }
    lapply(at, add_grid_line)
    
    xgrid = (1:12) - 0.5
    at = c(at,y_max * 0.9) 
    at_lab = max(at) + 0.05 * (max(at) - min(at))
    add_month_lab <- function(x, mnth) {
        tgrid = - 2 * pi * (x-3) / 12
        x_cart = at_lab * cos(tgrid)
        y_cart = at_lab * sin(tgrid)
        text(x_cart, y_cart, mnth, xpd = NA, srt = -90+360*tgrid/(2*pi))
        tgrid = - 2 * pi * (x-2.5) / 12
        x_cart1 = at_lab * cos(tgrid)
        y_cart1 = at_lab * sin(tgrid)
        lines(c(0, x_cart1), c(0, y_cart1), lty = 3, col = 'grey')
    }
    mapply(add_month_lab, xgrid, month.abb)
    
    years = unique(substr(time, 1, 4))
    cols = colorRampPalette(cols)(length(years))
    
    add_year <- function(year, col, BAi) {
        test = substr(time, 1, 4) == year
        x = as.numeric(substr(time[test], 6, 7))
        y = BAi[test]
        
        print(paste("year:", year, "\tval:", sum(y)))
        
        if (year > years[1] && FALSE) {
            x = c(x[1] - 1, x)
            y = c(BAi[which(test)[1]-1], y)
        }
        if (year < tail(years, 1)) {
            x = c(x, tail(x, 1) + 1)
            y = c(y, BAi[tail(which(test), 1)+1])
        }
        
        if (as.numeric(year) == target_year) {
            lwd = 2.5
            lty = 1
            print(which.max(y))
        } else {
            lwd = 1
            lty = 3
        }   
        theta = - 2 * pi * (x-3.5) / 12
    
        # polar → cartesian
        x_cart = y * cos(theta)
        y_cart = y * sin(theta)
        #lines(x_cart, y_cart, col = 'black', lwd = 0.5, xpd = NA)
        lines(x_cart, y_cart, col = col, lwd = lwd, lty = lty, xpd = NA)
    }
    
    mapply(add_year, years, 'black', MoreArgs = list(BAi = cBA))
    mapply(add_year, years, cols, MoreArgs = list(BAi = BA))
    return(list(cols, years))
}

png("figs/BA_Base_TS.png", height = 6, width = 6, units = 'in', res = 300)
layout(rbind(t(matrix(1:2, ncol = 1)), 3))
par( mar = rep(0.9, 4), oma = c(1.8, 1.2, 1.8, 1.2))
dat = read.csv('data/data/driving_data_base/Amazon/burnt_area_data.csv', 
               stringsAsFactors = FALSE)
plot_BA_type('Amazon', FALSE, '', 'Amazonia')#'All region')

#plot_BA_type('p95_burnt_area', 'p95_burnt_area_climateology', '', 'Sub-regional extremes')
#plot_BA_type('Amazon', TRUE, '', 'Sub-regional extremes')
#dat = read.csv('data/data/driving_data_base/Pantanal/burnt_area_data.csv', 
#                stringsAsFactors = FALSE)
#plot_BA_type('mean_burnt_area', 'mean_burnt_area_climateology', 'Pantanal', '')
cols = plot_BA_type('Pantanal', FALSE, '', 'Pantanal')
#cols = plot_BA_type('p95_burnt_area', 'p95_burnt_area_climateology', '', '')
#cols = plot_BA_type('Pantanal', TRUE, '', '')
years = cols[[2]]; cols = cols[[1]]
plot.new()
legs = unique(years)
lty = rep(3, length(legs))
lwd = rep(1, length(legs))
lty[legs == target_year] = 1
lwd[legs == target_year] = 3
legs = c(legs, 'Climateology')
lty = c(lty, 1)
lwd = c(lwd, 3)
cols = c(cols, 'black')
legend('top',legs , ncol = 3, col = cols, lty = lty, lwd = lwd)
dev.off()
