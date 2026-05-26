source("../rasterextrafuns/rasterPlotFunctions/R/make_col_vector.r")
graphics.off()
logit <- function(x) {
    x[x<0.0000000001] = 0.0000000001
    if (max(x) > 1) browser()
    log(x/(1-x))
}

rnning_mean <- function(r) 
    filter(r, rep(1 / 10, 10), sides = 1)

make_plot <- function(region, dir, pattern, file, burnt_area_event, percentiles) {
    
    percentile = percentiles[1]
    openDat <- function(exp) {
        print(exp)
        dirs = list.dirs(dir, full.name = TRUE, recursive=TRUE)
        dirs = dirs[grep(paste0('-', exp), dirs)]
        #browser()
        for (ptt in pattern) dirs = dirs[grep(paste0(ptt), dirs)]
        if (length(dirs) > 1) { 
            if (length(dirs) == 6) dirs = dirs[-1]
            #else dirs = dirs[!(grepl('mean', dirs) | grepl('pc-', dirs))]
            # (length(dirs) > 1) if (length(dirs) == 6) dirs = dirs[-1]
        }

        openDir <- function(dir) {   
            read.csv(paste0(dir, '/', file))
        }
        dat = lapply(dirs, openDir)
        return(dat)
    }
    dats = lapply(experiments, openDat)
    
    scale = burnt_area_event/quantile(as.matrix(dats[[1]][[1]]), percentile)
    dats = lapply(dats, lapply, function(i)  i[,seq(2, dim(i)[2], by = 3)])
    dats = lapply(dats, lapply, function(i)  i * scale)
    
    join_dats <- function(dat, id) { 
        out = mapply(cbind, dats[[id]], dat, SIMPLIFY = FALSE)
        return(out)        
        
    }
    dats[4:6] = lapply(dats[4:6], join_dats, 3)
    start[4:6] = start[3]

    years = mapply(function(x, st) st + (1:ncol(x[[1]])) - 1, dats, start)
    
    find_occurnaces <- function(pc) {
        print(pc)
        #if (grepl('Control', file)) 
        
        all = as.numeric(as.matrix(dats[[1]][[1]]))
        if (pc == 1) {
            test = logit(sort(unique(all)))
            nt = length(test)
            xt = seq(logit(1/nt), logit(1-1/nt), length.out = nt)
            event = predict(smooth.spline(test ~ xt, spar = 0.5), logit(1-1/(1.5*nt)))[[2]]
        } else 
            event = logit(quantile(all, pc))
        #else browser()
            #if (grepl('Canada', dir)) event = -3.389603
        find_occurnace <- function(dat) {
            for_run <- function(x) {
                find_pc <- function(r) {
                    
                    if (!grepl('Control', file)) return(0.01*mean(r/apply(dats[[1]][[1]], 1, mean)))
                    r0 = r
                    r = logit(unique(sort(r)))
                    
                    xr = seq(0, 1, length.out = length(r)+1)
                    xr = xr[-1] - diff(xr)/2
                    #xr = log(xr/(1-xr))
                    xp = seq(0, 1, length.out = 1000)
                    rp = predict(smooth.spline(r ~ xr, spar = 0.5), xp)[[2]]
                    rp = sort(rp)
                    #return(quantile(rp, 0.9986))
                    return(mean(event < rp))
                }
                out = apply(x, 2, find_pc)
                #browser()
                return(rnning_mean(out))
            }
            lapply(dat, for_run)
        }
        return(lapply(dats, find_occurnace))
    }
    freqs = lapply(percentiles, find_occurnaces)
    
    find_quatiles <- function(dat, year) { 
        for_run <- function(y) {                
            yq = apply(y, 1, rnning_mean)
            test = (year < 2019)
            #if (length(test) == 0) test = which(year == 2014)
            
            yq = apply(yq, 2, function(i) i / mean(i[test], na.rm = TRUE))
            apply(yq, 1, quantile, c(0.1, 0.9), na.rm = TRUE)
        }
        lapply(dat, for_run)
    }

    qdats = mapply(find_quatiles, dats, years, SIMPLIFY = FALSE)
    
    tfun <- function(x) return(100*x)
    add_Freq <- function(freq, pc, nme, ylab = '', axis1 = FALSE) {
        x_range = range(unlist(years, recursive = TRUE))
        y_range = range(tfun(unlist(freq, recursive = TRUE)), na.rm = TRUE)
        plot(x_range, y_range, type = 'n', xlab = '', ylab = '', axes = FALSE)
        mtext(nme, line = -1.5, side = 3)
        at = seq(1990, 2100, by = 10)
        axis(1, at = at, rep('', length(at)))
        if (axis1) axis(1, at = at)
        axis(2)
        mtext(ylab, side = 2, line = 2.5)
        add_experiment <- function(fq, year, col, alpha = '11') {
            add_run <- function(y) 
                lines(year, tfun(y), col = col, lty = 3, lwd = 0.5)
                
            if (length(fq) == 1) {
                lines(year, tfun(fq[[1]]), col = col, lwd = 2)
            } else {
                    
                pfq = apply(do.call(cbind, fq), 1, range)
                yearp = year[!is.na(pfq[1,])]
                pfq = pfq[,!is.na(pfq[1,])]
                if (nchar(col) == 7) colp = paste0(col, alpha) else colp = col
                polygon(c(yearp, rev(yearp)), tfun(c(pfq[1,], rev(pfq[2,]))), 
                        col = colp, border = NA)
                lines(yearp, tfun(pfq[1,]), col = col)
                lines(yearp, tfun(pfq[2,]), col = col)
                lapply(fq, add_run)
            }
            return(do.call(cbind, fq))
        }
        for (i in 1:4) mapply(add_experiment, freq, years, cols)
        fqs = mapply(add_experiment, freq, years, cols, '00')
    }
    png(paste0("futures_2", region, ".png"), res = 300, height = 7, width = 7.2, units = 'in')
    par(mfrow = c(3, 2), mar = c(3, 3, 0, 0), oma = c(1, 1, 0, 0))
    fqss = mapply(add_Freq, freqs, percentiles, percentile_names, 
            c('', '', '% Likelihood of event', rep('', 3)), c(rep(F, 4), T, T))

    legend('topleft', legend = c('historic', 'ssp126', 'ssp370', 'ssp585'), col = cols[-(2:3)], pch = 19, pt.cex = 2)
    dev.off()
    return(fqss)
}

regions = c('Canada', 'Greece', 'NW_Amazon')



experiments = c("factual", "counterfactual", "historical", "ssp126", "ssp370", "ssp585")
experiments = c("factual", "counterfactual", "historical", "ssp126", "ssp370", "ssp585")
start = c(2000, 2000, 1994, 2015, 2015, 2015)
#cols = c('#000000', '#0000FF', '#0000FF', '#0000FF', '#FF0000', '#FF0000', '#FF0000')#, '#FF0000')
cols = c('#000000', '#648FFF', '#DC267F00', '#785EF0', '#FFB000', '#FE6100')
cols = c('#000000', '#999999', '#DC267F00', '#7570b3', '#1b9e77', '#d95f02')

percentiles = c(0.5, 0.9, 0.95, 0.99, 0.999)
percentile_names = 1/(1-percentiles)
percentiles = c(NaN, percentiles)
percentile_names = c('2023 event', paste0('1-in-', round(percentile_names, 0)))



start = c(2000, 2000, -999, 1994, 1994, 1994)
pattern = "_13-frac_points_0.5-"
plot_region_fqi <- function(control_name, col_hint, region, pattern2, fi = 1) {
    dir = paste0("outputs/ConFire_", region, "-final/figs/")
    if (region == "Greece") dir = "outputs/ConFire_Greece-final/figs/"
    if (region == "Canada") dir = "outputs/ConFire_Canada-isimip-final/figs/"
    
    file = paste0("points-", control_name, ".csv")

    load(paste0("outputs/obs_time_series/", region, "/outs.Rd"))
    #if (percentile == 1) percentile = 1-1/120
    percentiles[1] = percentile
    
    fqss = make_plot(region, dir, c(pattern, pattern2), file, burnt_area_event, percentiles)
    fqs = fqss[,fi]
    
    years = seq(2019, 2099, by = 10)
    if(col_hint != "NULL") cols = sapply(cols, function(col) make_col_vector(c(col, col_hint), ncols = 3)[2])
    get_points_for_year <- function(year) {
        get_dat <- function(fq, st, id) {
            if (id ==2) return(NULL)
            if (is.null(dim(fq))) {
                yrs = st + 1:length(fq) - 1
                id = which(yrs == year)
                if (length(id) == 0) id = which(abs(yrs - year) == 1)
                if (length(id) == 0) return(NULL)
                out = fq[id] * 100
            } else {
                yrs = st + 1:(dim(fq)[1])
                id = which(yrs == year)
                if (length(id) == 0) id = which(abs(yrs - year) == 1)
                if (length(id) == 0) return(NULL)
                
                out = fq[id,] * 100
            }
            
            return(out)
        }
        out = mapply(get_dat, fqs, start, 1:length(fqs))
           
    }
    pnts = lapply(years, get_points_for_year)
    scale <- function(pnt) {
        for_mod <- function(i) {
            a = sort(pnts[[1]][[i]])
            b = sort(pnt[[i]])
            out = b/a
            if (!grepl('ontrol', control_name))  out = (out - 1)*100
            #else out[out>21.1] = 21.1
                return (out)
        }
        pnt[4:6] = lapply(4:6, for_mod)
        return(pnt)
    }
    if (region == 'Canada' && grepl('ontrol', control_name)) pnts[[1]][[5]][4] = 0.11 
    pnts = lapply(pnts, scale)
    
    #browser()
    #pnts = lapply(pnts, function(x) lapply(x, function(i) 100*(i-1)))
    plot_for_year <- function(pnt, year) {
        plot_point <- function(y, col, off) {
            #col = make_col_vector(c(col, col_hint), ncols = 3)[2]
            if (is.null(y)) return()      
            x = year + (off-5)*2 + c(-1, 1)
            if (length(y) == 1) {  
                if (grepl('ontrol', control_name))
                    lines(range(years), c(1, 1), col = col, lwd = 2, lty = 2)
                else 
                    lines(range(years), c(0, 0), col = col, lwd = 2, lty = 2)
            } else {
                yp = range(y)
                polygon(x[c(1, 2, 2, 1, 1)], yp[c(1, 1, 2, 2, 1)], 
                        col = paste0(col, '66'), border = paste0(col, 'BB'))
                lapply(y, function(i) lines(x, rep(i, 2), col = col))
            }
            return(col)
        }
        cols = mapply(plot_point, pnt, cols, 1:length(pnt))
    }

    yrange = range(unlist(lapply(pnts, function(i) i[4:6])))
    #browser()
    #if  (region == "NW_Amazon") browser()
    plot(c(2010, 2110), yrange, type = 'n', axes = FALSE, xlab = '', ylab = '')
    axis(1, at = years + 1, labels = rep('', length(years)))
    if (control_name == tail(controls, 1)) axis(1, at = years + 1)
    colsp = mapply(plot_for_year, pnts, years)
    
    if (pnts[[1]][[1]] !=0) scaler = pnts[[1]][[1]]  else scaler = 1
    yrange4 = yrange/scaler
    labels = round(seq(yrange4[1], yrange4[2], length.out = 6), 1)
    
    
    
    if (control_name == controls[1]) {
        #if (region == tail(regions, 1)) mtext(side = 4, 'Likelihood (%)', line = 3.5)
        if (region == regions[1]) mtext(side = 2, 'times more likely', line = 2.5)
        axis(2)
        if (region == "NW_Amazon") regionT = "Western Amazonia" else regionT = region
        mtext(side = 3, line = -1, regionT)
        #axis(2, at = labels * scaler, labels = labels)
    } else {
        axis(2)
    }
    if (region == regions[1]) {
        if (control_name == "Control") name2 = 'Burned Area'
            else name2 = sub("standard-", "", control_name)
        mtext(side = 2, line = 3.75, name2)
        legend('topleft', experiments[-(2:3)], col = paste0(cols[-(2:3)], 'BB'), pt.cex = 2, pch = 15, bty = 'n')
        legend('topleft', experiments[-(2:3)], col = cols[-(2:3)], pt.cex = 2, pch = 1, bty = 'n')
    }
    colate_tstep <- function(pnt)
        do.call(c, lapply(pnt, function(i) if (is.null(i)) return(NaN) else return(i)))
    out = sapply(pnts, colate_tstep)
    colnames(out) = years
    
    rname <- function(pnt, experiment) {
        if (is.null(pnt) || length(pnt) == 1) return(experiment)
        paste(experiment, 1:length(pnt))
    }
    rownames(out) = unlist(mapply(rname, pnts[[1]], experiments))
    write.csv(out, paste('figs/future_table', region, control_name, '-', fi, '.csv', sep = '-'))
}

controls = c('Control', 'standard-Fuel', 'standard-Moisture')#, 'standard-Ignition') 
cols_hint = c('NULL', '#00FF00', '#0000FF')#, '#FF0000')#, '#333333')
plot_fi <- function(fi) {
    pdf(paste0("figs/Figure_17box_futures3-", fi, "-.pdf"), 
        width = 8, height = 5.5) #res = 300, units = 'in', 
        par(mfcol = c(3, 3), oma = c(2, 4, 2, 4), mar = c(1, 2, 0, 2))
        pnts = mapply(function(region, pattern2) 
                        mapply(plot_region_fqi, controls, cols_hint, 
                               MoreArgs = list(region, pattern2, fi = fi)), regions, 
                                                c('pc-95', 'pc-95', 'pc-95'))

        #mtext(side = 4, 'Likelihood (%)', outer = TRUE, line = 2.5)
        #mtext(side = 2, 'times more likely', outer = TRUE, line = 2.5)
        mtext(side = 2, 'Control Stength (%)', outer = TRUE, line = 0.5, adj = 0.33)
    dev.off()
}

plot_fi(1)
#plot_fi(5)
