
openDat <- function(dir, region, subdirs, experiment, area, mnths, years) {
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

log10_plus <- function(x) (log10(x) + 1)/2



af_iscale <- function(x) x/(1-x)


regions = c("Midwestern Canadian Shield forests", 
            "Chilean Temperate Forests and Matorral", 
            "Northwest Iberia")

HadGEM_dir = "outputs/outputs_scratch/SoW2526/attribution-HadGEM-test29-fuelcf4/<<region>>/time_series/_16-frac_points_0.5/"
ISIMIP_dir = "outputs/outputs_scratch/SoW2526/isimip/full-4-notree/<<region>>/time_series/_15-frac_points_0.5/"

gcms = c("GFDL-ESM4-", "IPSL-CM6A-LR-", "MPI-ESM1-2-HR-", "MRI-ESM2-0-")#, "UKESM1-0-LL-")

region = tail(regions, 1)
mnths = c('01','02', '03', '04', '05', '06', '07','08', '09', '10', '11', '12')
mnths = c('08')
years = c(2025)

new_empty_plot_rr <- function(xlim = c(0, 2), ylab = 'Risk ratio', 
                              add_xlabs = ylab != '', ylim = NULL, ...) {
    if (is.null(ylim)) {
        ylim = c(0, 1)
    } else {
        ylim = log10_plus(ylim)
    }
    plot(xlim,  ylim, xlab = '', ylab = '', type = 'n', yaxt = 'n', yaxs = 'i', 
         xaxt = 'n', ...)
    labels = c(1/10, 1/5, 1/2, 1, 2, 5, 8, 10)
    
    at = log10_plus(labels)
    axis(2, at = at, labels = labels)
    mtext(side = 2, line = 3, ylab)
    for (y in at)
        lines(c(-9E9, 9E9), c(y, y), col= 'grey', lty = 2)
    return(ylim[1])
}

new_empty_plot_logit <- function(xlim = c(0, 2), ylab =  'Amplifcation factor',
                                 add_xlabs = ylab != '', ylim =  NULL, ...) {
    if (is.null(ylim)) {
        ylim = c(0, 1)
    } else {
        ylim = af_tscale(ylim)
    }
    plot(xlim,  ylim, xlab = '', ylab = '', type = 'n', yaxt = 'n', yaxs = 'i', 
         xaxt = 'n', ...)
    
    labels = c(0, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 2/3, 1, 1.5, 2, 4, 8, 16, 32, 64, 128, 1000000)
    at = af_tscale(labels)
    if (add_xlabs) 
        labels = c('0', '', '', '', '', '', '1/4', '1/2', '2/3', 'no\nchange', '1 1/2',
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
    return(ylim[1])
}


plot_af <- function(fact, cfact, xpos = 1, col = 'red', name = '', bar = TRUE, label = '', ...) {
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
    for (i in c(1, 3, 5)) 
        text(x = xpos, y = pcs[i], round(pc[i], 2), adj = c(-1, 0.5))
}




att_af_calc <- function(dir, region,factual_name, cfactual_name, exp, mnths, years,
                        BA = NULL, xp, xoffset, samples = NULL, plot_fun = plot_af, ...) {
    
    
    fact = openDat(dir, region, factual_name, exp, "mean", mnths, years)
    cfact = openDat(dir, region, cfactual_name, exp, "mean", mnths, years)
    

    if (is.null(samples)) {
        if (is.null(BA)) BA = openDat(dir, region, factual_name, "observation", "mean", mnths, years)
        prob = exp(BA*log(fact) + (1.0-BA)*log((1-fact)))
        samples = sample(1:length(prob), 1000, TRUE, prob)
    } else {
        samples = samples[[1]]
        BA = samples[[2]]
    }
    facts = fact[samples]
    cfacts = cfact[samples] 
    plot_fun(facts, cfacts, xp + xoffset, ...)
    return(list(samples, BA))
}


cumm_pdf <- function(x0, BA) {
    x = seq(0, 1, 0.0001)
    y = exp(BA*log(x) + (1.0-BA)*log((1-x)))
    y = cumsum(y)/sum(y)
    out = sapply(x0, function(xi) y[which(x>xi)[1]])
    
    return(out)
}

att_rr_calc <- function(dir, region, factual_name, cfactual_name, exp, mnths, years,
                        BA = NULL, xp, xoffset, samples = NULL, 
                        name = name, col = col, width = 0.2, ...) {
    
    xs = xoffset + xp + width*0.5*c(-1, 1)
    fact = openDat(dir, region, factual_name, exp, "mean", mnths, years)
    cfact = openDat(dir, region, cfactual_name, exp, "mean", mnths, years)
    if (is.null(samples)) {
        if (is.null(BA)) BA = openDat(dir, region, factual_name, "observation", "mean", mnths, years)
        prob1 = cumm_pdf(fact, BA)
        prob2 = cumm_pdf(cfact, BA)
        rr = sum(prob1)/sum(prob2)
        samples = list(list(prob1, prob2), BA)
    } else {
        BA = samples[[2]]
        samples = samples[[1]]
        rr = sum(fact* samples[[1]])/sum(cfact * samples[[2]])
    }
    lines(xs, rep(log10_plus(rr), 2), lwd = 3, col = col)
    
    text(x = xs[1], y = log10_plus(rr), round(rr, 2), adj = c(-1, 0.5))
    return(samples)
}

perm_test_paired <- function(d, transform = log, inverse = exp) {
    
    d = transform(d)
    
    obs = mean(d)
    signs <- expand.grid(rep(list(c(-1, 1)), length(d)))
    perm_means <- apply(signs, 1, function(s) mean(d * s))
    p_value <- mean(abs(perm_means) >= abs(obs))
    list(
        mean_difference = inverse(obs),
        median = inverse(quantile(obs, 0.5)),
        p_value = p_value,
        null_distribution = perm_means
    )
}


futr_af_calc <- function(dir, region, factual_name, cfactual_name, exp, mnths, years,
                         BA, xp, xoffset, samples = NULL, name = name, col = col, width = 0.05, 
                        ylim0 = 0, yearss = NULL, ...) {
    
    if (exp == "Evaluate") {
        if (factual_name[2] == "ssp370") {
            polygon(width/2 + c(xoffset, xoffset + 0.3)[c(1, 1, 2, 2)], c(0, 1, 1, 0),
                     col = '#00000011', border = NA)
            text(xp + xoffset, ylim0, adj = c(0.5, -0.3), 
                 paste0(range(years[[2]]), collapse = ' - '), font = 2)
        } else if (factual_name[2] == "ssp585")  {         
            polygon(width/2 + c(xoffset, xoffset + 0.3)[c(1, 1, 2, 2)], c(0, 1, 1, 0),
                     col = '#00000022', border = NA)
            lines(rep(width/2 + xoffset + 0.3, 2), c(-9E9, 9E9), lty = 2)
        }
       
    }
    
    if (years[[2]][1]== yearss[[1]][1] && exp == "Evaluate") 
        text(xoffset + xp/2, ylim0, adj = c(-1, 0.5), srt = 90, factual_name[2])
    
    xs = xoffset + xp/3 + width*0.5*c(-1, 1)
    for_gcm <- function(gcm) {
        fact = openDat(dir, region, paste0(factual_name, '/', gcm), exp, "mean", mnths, years[[1]])
        cfact = openDat(dir, region,  paste0(cfactual_name, '/', gcm), exp, "mean", mnths, years[[2]])
        qt =  mean(fact <= BA)
        
        af = quantile(cfact,qt)/quantile(fact,qt)
        lines(xs, rep(af_tscale(af), 2), lwd = 1, col = col)
        return(af)
    }
    afs = sapply(gcms, for_gcm)
    polygon(xs[c(1, 1, 2, 2)], af_tscale(range(afs)[c(1,2,2,1)]), 
            border = NA, col = paste0(col, "66")) 

    out = perm_test_paired(afs)
    text(x = xs[1], y = af_tscale(out$mean), adj = c(-0.67, 0.5), round(out$mean, 2))
    text(x = xs[1], y = af_tscale(max(afs)), adj = c(-0.67, 0.5), round(max(afs), 2))
    text(x = xs[1], y = af_tscale(min(afs)), adj = c(-0.67, 0.5), round(min(afs), 2))
    text(x = mean(xs), y = af_tscale(min(afs)), adj = c(0.5, 1), round(out$p_value, 2))
}

futr_rr_calc <- function(dir, region, factual_name, cfactual_name, exp, mnths, years,
                         BA, xp, xoffset, samples = NULL, name = name, col = col, width = 0.05, 
                         ylim0 = 0, yearss = NULL, ...) {
    
    if (exp == "Evaluate") {
        if (factual_name[2] == "ssp370") {
            polygon(width/2 + c(xoffset, xoffset + 0.3)[c(1, 1, 2, 2)], c(0, 1, 1, 0),
                     col = '#00000011', border = NA)
            text(xp + xoffset, ylim0, adj = c(0.5, -0.3), 
                 paste0(range(years[[2]]), collapse = ' - '), font = 2)
        } else if (factual_name[2] == "ssp585")  {         
            polygon(width/2 + c(xoffset, xoffset + 0.3)[c(1, 1, 2, 2)], c(0, 1, 1, 0),
                     col = '#00000022', border = NA)
            lines(rep(width/2 + xoffset + 0.3, 2), c(-9E9, 9E9), lty = 2)
            print(rep(width/2 + xoffset + 0.3, 2))
        }
       
    }
    if (years[[2]][1]== yearss[[1]][[2]][1] && exp == "Evaluate") 
        text(xoffset + xp/2, ylim0, adj = c(-1, 0.5), srt = 90, factual_name[2])
    
    xs = xoffset + xp/3 + width*0.5*c(-1, 1)
    for_gcm <- function(i) {
        gcm = gcms[[i]]
        fact = openDat(dir, region, paste0(factual_name, '/', gcm), exp, "mean", mnths, years[[1]])
        cfact = openDat(dir, region,  paste0(cfactual_name, '/', gcm), exp, "mean", mnths, years[[2]])
        
        if (is.null(samples) || exp == "Evaluate") {

            
            prob = exp(BA*log(fact) + (1.0-BA)*log((1-fact)))
            samples = sample(1:length(prob), 1000, TRUE, prob)
            facts = fact[samples]
            cfacts = cfact[samples]
            prob1 = cumm_pdf(facts, BA)
            prob2 = cumm_pdf(cfacts, BA)
            rr = sum(prob2)/sum(prob1)
            #rr = mean(cfact>BA)/mean(fact>BA)
            samples = list(prob1, prob2, samples)
            
            
        } else {        
            prob = samples[[i]]
            
            #rr = sum(prob[[1]])/sum(prob[[2]])*sum(cfact[prob[[3]]]* prob[[2]])/sum(fact[prob[[3]]] * prob[[1]])
            rr = mean(cfact[prob[[3]]]/fact[prob[[3]]])
        }
        print(rr)
        lines(xs, rep(log10_plus(rr), 2), lwd = 1, col = col)
        return(list(rr, samples))
    }
    outs = sapply(1:length(gcms), for_gcm)
    rrs = unlist(outs[1,])
    samples = outs[2,]   
    polygon(xs[c(1, 1, 2, 2)], log10_plus(range(rrs)[c(1,2,2,1)]), 
            border = NA, col = paste0(col, "66"))


    
    out = perm_test_paired(rrs)
    text(x = xs[1], y = log10_plus(out$mean), adj = c(-0.1, 0.5), round(out$mean, 2))
    text(x = xs[1], y = log10_plus(max(rrs)), adj = c(-0.1, 0.5), round(max(rrs), 2))
    text(x = xs[1], y = log10_plus(min(rrs)), adj = c(-0.1, 0.5), round(min(rrs), 2))
    text(x = mean(xs), y = log10_plus(min(rrs)), adj = c(0.5, 1), round(out$p_value, 2))
    return(samples)
}

plot_region <- function(region, HadGEM_dir, ISIMIP_dir, mnths, years,
                        empty_plot = new_empty_plot_logit, 
                        att_FUN = att_af_calc, futr_FUN = futr_af_calc,
                        ylim = NULL, reduced = TRUE) {
    
    add_run <- function(dir, factual_name = "factual-", cfactual_name = "counterfactual-",
                        xoffset = 0.0, years = 2025, mnths = NULL, 
                        plot_FUN = att_FUN, BA = 0.0, ylim0 = 0.0, ...) {
            
        add_experiemtnt <- function(exp = "Evaluate", col = 'red', xp = 0.25, 
                                    name = 'Burned Areas',
                                    samples = NULL) {
            
            samples = plot_FUN(dir, region, factual_name, cfactual_name, exp, mnths, years,
                        BA, xp, xoffset, samples, name = name, col = col, ylim0 = ylim0, ...)
            return(samples)
        } 
        samples = add_experiemtnt(col = "#E98400")
        exps = c("standard-Fuel", "standard-Moisture")#, "standard-Ignition","standard-Suppression")
        cols = c("#0096A1", "#EE0074")#, "purple", "grey")
        mapply(add_experiemtnt, exps, cols, c(0.5, 0.75), c("Fuel connectivity", "Dryness"),
               MoreArgs = list(samples = samples))
        
        return(samples[[2]])
    }
    
    ylim0 = empty_plot(xlim = c(0, 3 - 2*reduced), ylim = ylim)
    BA = add_run(HadGEM_dir, mnths = mnths, BA = NULL, ylim0 = ylim0)
    text(x = 0.5, y = ylim0, adj = c(0.5, -0.3), font = 2, 'HadGEM3-A\nfull ensemble')

    if (!reduced) {
        add_run(HadGEM_dir, mnths = mnths, cfactual_name = "counterfactual_mean-", 
                xoffset = 1, BA = BA, ylim0 = ylim0)
        add_run(ISIMIP_dir, xoffset = 2, years =  2002:2019, BA = BA, ylim0 = ylim0)        
        text(x = 1.5, y = ylim0, adj = c(0.5, -0.3), font = 2, 'HadGEM3-A\nensemble mean')
        text(x = 2.5, y = ylim0, adj = c(0.5, -0.3), font = 2, 'ISIMIP3a')
    }
    plot.new()

    if (reduced) {
        yearss = list(2030:2039, 2040:2049, 2090:2099)
    } else {
        yearss = list(2020:2029, 2030:2039, 2040:2049, 2050:2059, 2060:2069, 
                      2070:2079, 2080:2089, 2090:2099)
    }
    ylim0 = empty_plot(xlim = c(0, length(yearss)-0.075), ylab = '', xaxs = 'i', ylim = ylim)
    for_ssp <- function(ssp, xmini_off, yearss) {
        subdir = c("historical", ssp)
        
        for_yrss <- function(yrss, xoffset) {
            
            add_run(ISIMIP_dir, xoffset = xoffset-1 + xmini_off, years =  list(2010:2019,yrss), 
                    mnths = mnths,yearss = yearss,
                    plot_FUN = futr_FUN, factual_name = subdir, cfactual_name = subdir, 
                    BA = BA, ylim0 = ylim0)
        }
        mapply(for_yrss, yearss, 1:length(yearss))
    }
    mapply(for_ssp, c("ssp126", "ssp370", "ssp585"), c(0, 0.3, 0.6), MoreArgs = list(yearss))
    
}



plot_region_all_plots <- function(region, mnths, years, ylim1 = NULL, ylim2 = NULL, reduced = TRUE) {
    png(paste0("figs/att_outlook", region, ".png"), width = 14 - 7*reduced, height = 9, units = 'in', res = 300)
    layout(rbind(1:3, 4:6), widths = c(0.2, 0.02, 0.4))
    par(oma = c(2, 5, 2, 5), mar = c(1, 0, 1, 0))
        plot_region(region,  HadGEM_dir, ISIMIP_dir, mnths, years, 
                     ylim = ylim1)
        plot_region(region,  HadGEM_dir, ISIMIP_dir, mnths, years, 
                    empty_plot = new_empty_plot_rr,
                    att_FUN = att_rr_calc, futr_FUN = futr_rr_calc, ylim = ylim2, reduced = reduced)
    dev.off()
}
regions = c("Northwest Iberia", "Midwestern Canadian Shield forests", "Chilean Temperate Forests and Matorral", "Scottish_Highlands", "Southeast_South_Korea")
list(c('08'), c('07', '08'), c('01', '02'), c('06', '07'), c('03'))
years = list(2025, 2025, 2026, 2025, 2025)
ylim1 = list(c(0.5, 9E99), NULL, NULL, NULL)
ylim2 = list(c(0.9, 3), NULL, NULL, NULL)
mapply(plot_region_all_plots,regions, mnths, years, ylim1, ylim2)
