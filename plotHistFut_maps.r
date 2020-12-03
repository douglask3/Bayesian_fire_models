library(raster)
library(rasterExtras)
library(gitBasedProjects)
library(ncdf4)
source("libs/plotStandardMap.r")
source("libs/sd.raster.r")
source("libs/filename.noPath.r")
graphics.off()

dir = 'outputs/sampled_posterior_ConFire_ISIMIP_solutions/attempt2/'

models = c("GFDL-ESM2M", "HADGEM2-ES", "IPSL-CM5A-LR", "MIROC5")

periods = c("historic", "RCP2.6", "RCP6.0")

variable = "burnt_area_mean"

limits =  c(0,1, 2, 5, 10, 20, 40)
dlimits = c(-10, -5, -2, -1, -0.1, 0.1,  1, 2, 5, 10)
cols = c('#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026')

dcols = rev(c('#a50026','#d73027','#f46d43','#fdae61','#fee090','#ffffbf','#e0f3f8','#abd9e9','#74add1','#4575b4','#313695'))

sc = 12 * 100

obs = mean(brick("data/ISIMIP_data/burnt_area_GFED4sObs.nc"))


logit <- function(r) {
    r[r < 0.0000001] = 0.0000001
    log(r/(1-r))
}

logistic <- function(r) 
    1/(1+exp(r*(-1)))


obsL = logit(obs)

plotMap <- function(model, period, cols, limits, dcols = NULL, dlimits = NULL, dat0 = NULL,
                    anomolise = NULL) {
    file = paste(dir, model, period, "model_summary.nc", sep = '/')
    dat = brick(file, varname = variable)[[c(5, 50, 95)]]
    
    if (!is.null(anomolise)) 
        if (is.null(dat0)) datP = obs else {        
            dat = logistic(logit(dat) - anomolise)
            datP = dat
    } else datP = dat
    plotStandardMap(datP * sc, cols = cols,limits = limits)
    if (model == models[1]) mtext(side = 3, period)
    if (is.null(dat0)) {
        mtext(side = 2, adj = 0.2, model)
        plot.new()
    } else {
        dat = (dat-dat0) * sc
        plotStandardMap(dat, cols = dcols,limits = dlimits)
    }
    return(dat)
}

legendFun <- function(cols, limits, dat) 
    add_raster_legend2(cols, limits, dat = dat,
                           transpose = FALSE, srt = 0, oneSideLabels= TRUE,
                           plot_loc = c(0.1, 0.9, 0.73, 0.8), ylabposScling=0.8, add = FALSE)

plotFun <- function(fname, anomolise = FALSE) {
    print(fname)
    print(variable)
    print("---")
    png(paste0("figs/", fname, "anomIs_", anomolise, "_Ymaps.png"),
        height = 10, width = 7.2, units = 'in',res  = 300)
        layout(rbind(matrix(1:(6*length(models)), ncol = 3), (6*length(models)) + c(1, 2, 2)))
        par(mar = rep(0, 4), oma = c(0, 2, 2, 0))
        
        if (anomolise) datA = obs else datA = NULL
        dat0 = lapply(models, plotMap, periods[1], cols, limits, anomolise = datA)
        if (anomolise) {
            datA = lapply(dat0, function(i) logit(i) - logit(obs))
            dat0 = list(obs)
        } else datA = list(NULL)
        datP = lapply(periods[2:3], function(p)
                    mapply(plotMap, models, dat0 = dat0, anomolise = datA,
                    MoreArgs = list(p, cols, limits, dcols, dlimits)))
        legendFun( cols,  limits, dat0[[1]])
        legendFun(dcols, dlimits, datP[[1]][[1]])

        
    dev.off.gitWatermark()

    triangulaise <- function(p, addLegend, lab) {
        perctile <- function(i) {
            
            q = layer.apply(p, function(q) q[[i]])
            P = mean(layer.apply(q, function(i) {i[i<0] = 0; i}))
            N = abs(mean(layer.apply(q, function(i) {i[i>0] = 0; i})))
            
            PNcols = make_col_vector(dcols, ncols = length( dlimits)+1)
            
            PNlims = dlimits[dlimits >0]
            P = cut_results(P, PNlims)
            N = cut_results(N, PNlims)
            PN = P + (N - 1) * (length(PNlims) + 1)
            
            k = 0
            PNcols = c()
            cols = make_col_vector(c(Ncols, Pcols), ncols = 101)
            index = seq(0, 1, length.out = length(PNlims)+1)
            legXY = c()
            for (i in index) for (j in index) {
                if (i == 0 && j == 0) rt = 50 else rt = round(100*j/(j+i))
                
                col = col0 = cols[rt+1]
                mag = round(100*(sqrt(i^2 + j^2)^0.5))
                mag[mag > 100] = 100
                col = make_col_vector(c("white", col), ncols = 101)[mag+1]
                #if (is.na(col)) browser()
                PNcols = c(PNcols, col)
                legXY = rbind(legXY, c(i, j))
            }            
            
            plotStandardMap(PN, cols = PNcols, limits = 0.5+1:((length(PNlims)+1)^2-1), 
                            readyCut = TRUE, speedy = FALSE)
            mtext(lab, side = 2, line = -2)
            if (addLegend) {
                par(mar = c(3, 13, 1, 11))
                plot(c(0, 1), c(0, 1), axes = FALSE, xlab = '', ylab ='', xaxs = 'i', yaxs = 'i')
                for (cex in 10*seq(10, 0, by = -0.2))
                    points(legXY[,1], legXY[,2], pch = 19, cex = cex, col = PNcols)
                polygon(c(0, 1, 1, 0), c(1, 1, 0, 1), col = "white", border = "white")
                mtext("Decrease (%)", side = 1, line = 2)
                mtext("Increase (%)", side = 2, line = 2)
                lapply(1:2, axis, at = index - c(0, diff(index)/2), labels = c(0, PNlims))
            
            }
    
        }
        #layer.apply(1:3, perctile)
        perctile(2)
    }
    png(paste0("figs/", fname, "anomIs_", anomolise, "_CXmaps.png"),
        height = 10, width = 7.2, units = 'in',res  = 300)
    layout(matrix(1:5, ncol = 1), heights = c(1, 0.3, 1, 1, 1))
    par(mar = rep(0, 4))
    if (length(dat0) > 1) dat0 = mean(layer.apply(dat0, function(i) i[[2]])) else dat0 = dat0[[1]]
    
    plotStandardMap(dat0 * sc, cols = cols, limits = limits, speedy = FALSE)
    mtext(side = 2, "Historic", line = -1)
    legendFun( cols,  limits, dat0[[1]])
    mapply(triangulaise, datP, c(F, T), c("RCP2.6", "RCP6.0"))
    dev.off()
    
}
Pcols = "#330000"
Ncols = "#000033"

plotFun("BA")

limits =  c(0, 0.1, 0.5, 1, 5, 10, 50)
dlimits = c(-10, -5, -1, -0.5, -0.1, -0.05, -0.01, 0.01, 0.05, 0.1, 0.5, 1, 5, 10)

plotFun("BA", TRUE)


plotCOntrols <- function(control) {
    variable <<- paste0("standard_",control)
    sc <<-  100# * 10
    limits <<- c(0, 1, 2, 5, 10, 20, 50, 100, 150)/10
    limits <<- c(-100, -50, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 50, 100)/10
    plotFun(variable)

    sc <<-  100 * 12
    variable <<- paste0("potential_",control)
    limits <<- c(0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15)
    dlimits <<-  c(-0.2, -0.1, -0.01, -0.001, 0.001, 0.01, 0.1, 0.2)/10
    plotFun(variable)

    variable <<- paste0("sensitivity_",control)
    
    sc <<- 100 * 4 * 12
    #limits <<- c(0, 1, 2, 5, 10, 20, 50, 100, 150)/10
    #limits <<- c(-100, -50, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 50, 100)/10
    plotFun(variable)    
}

cols = c('#ffffe5','#f7fcb9','#d9f0a3','#addd8e','#78c679','#41ab5d','#238443','#006837','#004529')
dcols = c('#40004b','#762a83','#9970ab','#c2a5cf','#e7d4e8','#f7f7f7','#d9f0d3','#a6dba0','#5aae61','#1b7837','#00441b')

Pcols = "#110011"
Ncols = "#001111"

plotCOntrols("fuel")


cols = c('#ffffe5','#fff7bc','#fee391','#fec44f','#fe9929','#ec7014','#cc4c02','#993404','#662506')

dcols = rev(c('#543005','#8c510a','#bf812d','#dfc27d','#f6e8c3','#f5f5f5','#c7eae5','#80cdc1','#35978f','#01665e','#003c30'))

Pcols = "#190900"
Ncols = "#001909"

plotCOntrols("moisture")