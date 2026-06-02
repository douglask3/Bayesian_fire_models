
openDat <- function(dir, subdirs, experiment, area, mnths, years) {
    openFile <- function(subdir) {
        file = gsub('<<region>>', gsub(' ', '_', region), dir)
        file = paste(file, subdir, area, '/members/absolute/', 
                     paste0(experiment, '.csv'),
                     sep = '/')
        print(file)
        return(read.csv(file))
    }
    dats = lapply(subdirs, openFile)
    dat = do.call(cbind, dats)
    
       
    in_time = apply(sapply(years, function(yr)
                        substr(colnames(dat), 2, 5) == as.character(yr)), 1, any)    

    if (!is.null(mnths)) {
        in_mnth = apply(sapply(mnths, function(mn)
                        substr(colnames(dat), 7, 8) == as.character(mn)), 1, any)
        in_time = in_time & in_mnth
    }
    out = dat[,in_time]
    
    if (sum(in_time) > 1) {
        if (is.null(mnths) || length(mnths) > 1) {
            cout = substr(colnames(out), 2, 5)
            yrs = unique(cout)
            day = lapply(yrs, function(yr) apply(out[,cout == yr], 1, mean)) 
        }
        out = unlist(out)
    }  
    return(out)
}

kde <- function(x) {
    dens <- density(x)

    approx_tail <- with(dens, {
        idx <- which(x >= BA)
        if (length(idx) == 0) return(0)
        sum(diff(x)[idx[-length(idx)]] * y[idx[-1]])
    })
}

af_tscale <- function(x) {
    y = x
    y[x>=1] = 1-1/x[x>=1]
    y[x<1] = -(1-x[x<1])
    return((y+1)/2)
}
af_iscale <- function(x) x/(1-x)

plot_af <- function(fact, cfact, xpos = 1, col = 'red', name = '', bar = TRUE, label = '') {
    af = fact/cfact
    if (bar) {
        pc = quantile(af, c(0.05, 0.25, 0.5, 0.75, 0.95))
        outline = af_tscale(range(af))
        
        print(pc-1)
        pcs = af_tscale(pc)
        for (i in c(1, 5))
            lines(xpos + c(-0.05, 0.05), rep(pcs[i], 2), col = col)
        lines(rep(xpos, 2), pcs[c(1, 5)], col = col)
        polygon(xpos + 0.1*c(-1, -1, 1, 1), pcs[c(2, 4, 4, 2)], border = NA, col = col) 
        lines(xpos + 0.1*c(-1, 1), rep(pcs[3], 2), lwd = 2, xpd = NA)
        points(rep(xpos, 2), outline, col = col)
        
    }
    likelihood = round(mean(af>1)*100 + mean(af==1)*50)
    #text(x = xpos - 0.1, y = pcs[3], paste0(likelihood, '%'), adj = 1.1) 
    text(x = xpos, y = pcs[1], paste0(likelihood, '%'), adj = c(0.5, 2)) 
    #text(x = xpos, y = min(pcs), adj = 1.1, srt = 45, name)
}

regions = c("Midwestern Canadian Shield forests", 
            "Chilean Temperate Forests and Matorral", 
            "Northwest Iberia")

HadGEM_dir = "outputs/outputs_scratch/SoW2526/attribution-HadGEM-test29-fuelcf4/<<region>>/time_series/_16-frac_points_0.5/"
ISIMIP_dir = "outputs/outputs_scratch/SoW2526/isimip/full-4-notree/<<region>>/time_series/_15-frac_points_0.5/"

gcms = c("GFDL-ESM4-", "IPSL-CM6A-LR-", "MPI-ESM1-2-HR-", "MRI-ESM2-0-", "UKESM1-0-LL-")

region = tail(regions, 1)
mnths = c('01','02', '03', '04', '05', '06', '07','08', '09', '10', '11', '12')
mnths = c('09')
years = c(2025)


new_empty_plot_logit <- function(xlim = c(0, 2), ylab =  'Amplifcation factor',
                                 add_xlabs = ylab != '', ...) {
    plot(xlim,  c(0, 1), xlab = '', ylab = '', type = 'n', yaxt = 'n', yaxs = 'i', 
         xaxt = 'n', ...)
    
    labels = c(0, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64, 128, 1000000)
    at = af_tscale(labels)
    if (add_xlabs) 
        labels = c('0', '', '', '', '', '', '1/4', '1/2', 'no\nchange', 
                   '2', '4', '', '', '', '', '', 'All from\nclimate')
    else
        labels = rep('', length(labels))
    axis(2, at = at, labels = labels)
    mtext(side = 2, line = 3, ylab)
    for (y in at)
        lines(c(-9E9, 9E9), c(y, y), col= 'grey', lty = 2)
    
    if (ylab == '') {   
        mtext(side =4, line = 3, "% explained by climate change")
        axis(4, seq(0, 1, length.out=  9), seq(-100, 100,length.out=  9))
    }
}

att_af_calc <- function(dir, factual_name, cfactual_name, exp, mnths, years,
                        BA, xp, xoffset, samples = NULL, ...) {
    
    fact = openDat(dir, factual_name, exp, "mean", mnths, years)
    cfact = openDat(dir, cfactual_name, exp, "mean", mnths, years)
    BA = 0.1
        
    if (is.null(samples)) {
        prob = exp(BA*log(fact) + (1.0-BA)*log((1-fact)))
        samples = sample(1:length(prob), 1000, TRUE, prob)
    }
    facts = fact[samples]
    cfacts = cfact[samples] 
    plot_af(facts, cfacts, xp + xoffset, ...)
    return(samples)
}

futr_af_calc <- function(dir, factual_name, cfactual_name, exp, mnths, years,
                         BA, xp, xoffset, samples = NULL, name = name, col = col, width = 0.05, 
                        ...) {
    
    if (exp == "Evaluate") {
        if (factual_name[2] == "ssp370") {
            polygon(width/2 + c(xoffset, xoffset + 0.3)[c(1, 1, 2, 2)], c(0, 1, 1, 0),
                     col = '#00000011', border = NA)
            text(xp + xoffset, 0, adj = c(0.5, -0.3), 
                 paste0(range(years[[2]]), collapse = ' - '), font = 2)
        } else if (factual_name[2] == "ssp585")  {         
            polygon(width/2 + c(xoffset, xoffset + 0.3)[c(1, 1, 2, 2)], c(0, 1, 1, 0),
                     col = '#00000022', border = NA)
            lines(rep(width/2 + xoffset + 0.3, 2), c(-9E9, 9E9), lty = 2)
        }
       
    }
    if (years[[2]][1]== yearss[[1]][[2]][1] && exp == "Evaluate") 
        text(xoffset + xp/2, 0, adj = c(-1, 0.5), srt = 90, factual_name[2])
    
    xs = xoffset + xp/3 + width*0.5*c(-1, 1)
    for_gcm <- function(gcm) {
        fact = openDat(dir, paste0(factual_name, '/', gcm), exp, "mean", mnths, years[[1]])
        cfact = openDat(dir,  paste0(cfactual_name, '/', gcm), exp, "mean", mnths, years[[2]])
        qt =  mean(fact <= BA)
        
        af = quantile(cfact,qt)/quantile(fact,qt)
        lines(xs, rep(af_tscale(af), 2), lwd = 1, col = col)
        return(af)
    }
    afs = sapply(gcms, for_gcm)
    polygon(xs[c(1, 1, 2, 2)], af_tscale(range(afs)[c(1,2,2,1)]), 
            border = NA, col = paste0(col, "66"))
    #browser()
    
    
}
plot_region <- function(region, HadGEM_dir, ISIMIP_dir, mnths, years, yearss) {
    
    add_run <- function(dir, factual_name = "factual-", cfactual_name = "counterfactual-",
                        xoffset = 0.0, years = 2025, mnths = NULL, 
                        plot_FUN = att_af_calc) {
        
        add_experiemtnt <- function(exp = "Evaluate", col = 'red', xp = 0.25, 
                                    name = 'Burned Areas',
                                    samples = NULL) {
            
            samples = plot_FUN(dir, factual_name, cfactual_name, exp, mnths, years,
                        BA, xp, xoffset, samples, name = name, col = col)
            return(samples)
        } 
        samples = add_experiemtnt(col = "#E98400")
        exps = c("standard-Fuel", "standard-Moisture")#, "standard-Ignition","standard-Suppression")
        cols = c("#0096A1", "#EE0074")#, "purple", "grey")
        mapply(add_experiemtnt, exps, cols, c(0.5, 0.75), c("Fuel connectivity", "Dryness"),
               MoreArgs = list(samples = samples))
    }
    layout(rbind(1:3, 4:6), widths = c(0.2, 0.02, 0.8))
    par(oma = c(2, 5, 2, 5), mar = c(1, 0, 1, 0))
    new_empty_plot_logit(xlim = c(0, 3))
    add_run(HadGEM_dir, mnths = mnths)
    add_run(HadGEM_dir, mnths = mnths, cfactual_name = "counterfactual_mean-", xoffset = 1)
    add_run(ISIMIP_dir, xoffset = 2, years =  2002:2019)
    text(x = 0.5, y = 0, adj = c(0.5, -0.3), font = 2, 'HadGEM3-A\nfull ensemble')
    text(x = 1.5, y = 0, adj = c(0.5, -0.3), font = 2, 'HadGEM3-A\nensemble mean')
    text(x = 2.5, y = 0, adj = c(0.5, -0.3), font = 2, 'ISIMIP3a')
    plot.new()
    new_empty_plot_logit(xlim = c(0, length(yearss)), ylab = '', xaxs = 'i')
    for_ssp <- function(ssp, xmini_off) {
        subdir = c("historical", ssp)
        
        for_yrss <- function(yrss, xoffset) {
            add_run(ISIMIP_dir, xoffset = xoffset-1 + xmini_off, years =  list(2010:2019,yrss), 
                    mnths = mnths,
                    plot_FUN = futr_af_calc, factual_name = subdir, cfactual_name = subdir)
        }
        mapply(for_yrss, yearss, 1:length(yearss))
    }
    mapply(for_ssp, c("ssp126", "ssp370", "ssp585"), c(0, 0.3, 0.6))
    browser()
}

yearss = list(2020:2029, 2030:2039, 2040:2049, 2050:2059, 2060:2069, 
                      2070:2079, 2080:2089, 2090:2099)
yearss = list(2030:2039, 2040:2049, 2090:2099)
plot_region(region,  HadGEM_dir, ISIMIP_dir, mnths, years, yearss = yearss)
