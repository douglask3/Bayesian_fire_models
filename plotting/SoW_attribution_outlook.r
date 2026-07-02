graphics.off()
set.seed(123)

##########################################################
## librarys                                             ##
##########################################################
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
            out = lapply(yrs, function(yr) apply(out[,cout == yr], 1, mean)) 
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

af_mit_tscale <- function(x) {
    x = x /100
    y= x
    y[x>=0]= 1-0.5/(x[x>=0]+1)
    y[x<0]= 1-(1-0.5/(-x[x<0]+1))
    return(y)
}

log10_plus <- function(x) (log10(x) + 1)/2

cumm_pdf <- function(x0, BA) {
    x = seq(0, 1, 0.0001)
    y = exp(BA*log(x) + (1.0-BA)*log((1-x)))
    y = cumsum(y)/sum(y)

    find_inbetween <- function(xi) {
        p1 = tail(which(x <= xi), 1)
        p2 = which(x > xi)[1]
        (y[p2] *(xi-x[p1]) + y[p1] *(x[p2]-xi))/((xi-x[p1]) + (x[p2]-xi))
    }
    out = sapply(x0, find_inbetween)
    
    return(out)
}


perm_test_paired <- function(d, transform = log, inverse = exp) {
    
    d = transform(d)
    
    obs = mean(d)
    signs <- expand.grid(rep(list(c(-1, 1)), length(d)))
    perm_means <- apply(signs, 1, function(s) mean(d * s))
    p_value <- mean(abs(perm_means) >= abs(obs))
    if (obs < 0) {
        p_value = 100*p_value/2
    } else {
        p_value = 100-100*p_value/2
    }
    list(
        mean_difference = inverse(obs),
        median = inverse(quantile(obs, 0.5)),
        p_value = round(p_value),
        null_distribution = perm_means
    )
}

##########################################################
## defineing empty plots                               ##
##########################################################

new_empty_plot <- function(y_tfun, labels, labels_txt, 
                                xlim = c(0, 2), ylab =  'Amplifcation factor',
                                add_xlabs = ylab != '', ylim =  NULL, ...) {
    if (is.null(ylim)) {
        ylim = c(0, 1)
    } else {
        ylim = y_tfun(ylim)
    }

    plot(xlim,  ylim, xlab = '', ylab = '', type = 'n', yaxt = 'n', yaxs = 'i', 
         xaxt = 'n', ...)

    at = y_tfun(labels)
    
    #if (!add_xlabs) labels_txt[] = ''
    axis(2, at = at, labels = labels_txt)

    mtext(side = 2, line = 3, ylab)
    for (y in at)
        lines(c(-9E9, 9E9), c(y, y), col= 'grey', lty = 2)
    return(ylim[1])
}

new_empty_plot_rr <- function(...,  mitigate = False, ylab = 'Probability Ratio') {

    labels = c(1/10, 1/5, 1/2, 1, 2, 5, 8, 10)
    labels_txt = c('1/10', '1/5', '1/2', '1', '2', '5', '8', '10')
    new_empty_plot(log10_plus, labels, labels_txt, ylab = ylab, ...)
}

new_empty_plot_logit <- function(..., mitigate = FALSE, ylim = NULL) {
    labels = c(0, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 2/3, 1, 
               1.5, 2, 4, 8, 16, 32, 64, 128, 1000000)
    labels_txt = c('0', '', '', '', '', '', '1/4', '1/2', '2/3', 'no\nchange', '1 1/2',
                   '2', '4', '', '', '', '', '', 'All from\nclimate')
    
    if (!is.null(ylim) && sum(((labels) > ylim[1]) & ((labels) < ylim[2]))<5) {
        labels = c(0, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 2/3, 4/5, 1, 1.25,
               1.5, 2, 4, 8, 16, 32, 64, 128, 1000000)
        labels_txt = c('0', '', '', '', '', '', '1/4', '1/2', '2/3', '4/5', 'no\nchange', '1 1/5', '1 1/2',
                   '2', '4', '', '', '', '', '', 'All from\nclimate')
    }
    out = new_empty_plot(af_tscale, labels, labels_txt, ylim = ylim,...)
    
    return(out)
}


new_empty_plot_mitigate <- function(..., mitigate = FALSE, ylim = NULL) {
    
    labels = c(-500, -200, -100, -50, -20, -10, -5, -2, -1, 0, 
                1, 2, 5, 10, 20, 50, 100, 200, 500)
    
    
    if (!is.null(ylim) && sum(((labels) > ylim[1]) & ((labels) < ylim[2]))<5) 
        labels = labels/4
    
    out = new_empty_plot(af_mit_tscale, labels, labels, ylim = ylim,...)
    
    return(out)
}

##########################################################
## plotting functions                                   ##
##########################################################

plot_af <- function(af, xpos = 1, col = 'red', name = '', bar = TRUE, 
                    FUN = af_tscale, label = '', 
                     csv_out = NULL, csv_out_name = 'AF', bwidth = 0.1, ...) {
    bwidth_bar = 0.05*bwidth^(0.33)/0.1^(0.33)
    
    if (bar) {
        pc = quantile(af, c(0.05, 0.25, 0.5, 0.75, 0.95), na.rm = TRUE)
        outline = FUN(range(af, na.rm = TRUE))
        
        print(pc-1)
        pcs = FUN(pc)
        for (i in c(1, 5))
            lines(xpos + c(-bwidth_bar, bwidth_bar), rep(pcs[i], 2), col = col)
        lines(rep(xpos, 2), pcs[c(1, 5)], col = col)
        polygon(xpos + bwidth*c(-1, -1, 1, 1), pcs[c(2, 4, 4, 2)], border = NA, col = col) 
        lines(xpos + bwidth*c(-1, 1), rep(pcs[3], 2), lwd = 2, xpd = NA)        
    }
    likelihood = round(mean(af>1, na.rm = T)*100 + mean(af==1, na.rm = T)*50)
    
    out = cbind(name, csv_out_name, names(pc), round(pc, 2))
    out = rbind(out, c(name, 'likelihood', '%', likelihood))
    if (!is.null(csv_out)) 
            write.table(out, file = csv_out, sep = ",", 
                        append = TRUE, col.names = FALSE, row.names = FALSE)
}

## Amplifcation Factor
att_af_calc <- function(dir, region,factual_name, cfactual_name, exp, mnths, years,
                        BA = NULL, xp, xoffset, samples = NULL, ...) {
    
    fact = openDat(dir, region, factual_name, exp, cell_sample, mnths, years)
    cfact = openDat(dir, region, cfactual_name, exp, cell_sample, mnths, years)

    if (is.null(samples)) {
        if (is.null(BA))
            if (background_BA)
                BA = 0
            else
                BA = openDat(dir, region, factual_name, "observation", 
                             cell_sample, mnths, years)
        prob = exp(BA*log(fact) + (1.0-BA)*log((1-fact)))
        samples = sample(1:length(prob), 1000, TRUE, prob)
    } else {
        samples = samples[[1]]
        BA = samples[[2]]
    }

    if (background_BA) { 
        fact = sort(fact) + 0.0000001
        cfact = sort(cfact) + 0.0000001
    } else {
        fact = fact[samples]
        cfact = cfact[samples] 
    }
    plot_af(fact/cfact, xp + xoffset, ...)
    return(list(samples, BA))
}


att_rr_calc <- function(dir, region, factual_name, cfactual_name, exp, mnths, years,
                        BA = NULL, xp, xoffset, samples = NULL, ...) {
    
    fact = openDat(dir, region, factual_name, exp, cell_sample, mnths, years) 
    cfact = openDat(dir, region, cfactual_name, exp, cell_sample, mnths, years)
    if (is.null(samples)) {
        if (background_BA)
            BA = quantile(fact, 0.5)
        else
            if (is.null(BA))
                BA = openDat(dir, region, factual_name, "observation", 
                            cell_sample, mnths, years)
        prob1 = cumm_pdf(fact, BA)
        prob2 = cumm_pdf(cfact, BA)
        
        rr = prob1/prob2
        samples = list(list(prob1, prob2), BA)
    } else {
        samples0 = samples
        BA = samples[[2]]
        samples = samples[[1]]
        rr = (fact* samples[[1]])/(cfact * samples[[2]])
    }
    
    plot_af(rr, xp + xoffset, csv_out_name = 'rr', ...)
    
    
    return(samples)
}

futr_annotaion <- function(exp, factual_name, xp, xoffset, width, ylim0, years, yearss,
                           addSSPlab = TRUE)  {
    if (exp == BA_varname) {
        if (length(factual_name) > 1) factual_name = factual_name[2]
        fname_test = substr(factual_name, nchar(factual_name)-5, nchar(factual_name))
        if (fname_test == "ssp370" || fname_test == 'igated') {
            polygon(width/2 + c(xoffset, xoffset + 0.3)[c(1, 1, 2, 2)], c(0, 1, 1, 0),
                     col = '#00000011', border = NA)
            text(xp + xoffset, ylim0, adj = c(0.5, 1.3), xpd = NA,
                 paste0(range(years[[2]]), collapse = ' - '), font = 2)
        } else if (fname_test == "ssp585")  {         
            polygon(width/2 + c(xoffset, xoffset + 0.3)[c(1, 1, 2, 2)], c(0, 1, 1, 0),
                     col = '#00000022', border = NA)
            lines(rep(width/2 + xoffset + 0.3, 2), c(-9E9, 9E9), lty = 2)
        }
       
    }
 
    if (years[[2]][1]== yearss[[1]][1] && exp == BA_varname &&  addSSPlab) {
        yrange = par("usr")[3:4]
        ywhich = which.max(abs(yrange-0.5))
        if (ywhich == 2) {
            yp = yrange[2]
            adj = 1.1
        } else {
            yp = yrange[1]
            adj = -0.1
        }
         text(xoffset + xp/2, yp, adj = c(adj, 0.5), srt = 90, factual_name)
    }
    
}

futr_af_calc <- function(dir, region, factual_name, cfactual_name, exp, mnths, years,
                         BA, xp, xoffset, samples = NULL, name = name, col = col, width = 0.05, 
                        ylim0 = 0, yearss = NULL, ...) {
    
    futr_annotaion(exp, factual_name, xp, xoffset,width, ylim0, years, yearss)
    
    xs = xoffset + xp/3 + width*0.5*c(-1, 1)
    for_gcm <- function(gcm, i) {
        fact = openDat(dir, region, paste0(factual_name, '/', gcm), 
                       exp, cell_sample, mnths, years[[1]])
        cfact = openDat(dir, region,  paste0(cfactual_name, '/', gcm),
                        exp, cell_sample, mnths, years[[2]])
        
        if (!background_BA)  {
            if (is.null(samples)) {
                prob = exp(BA*log(fact) + (1.0-BA)*log((1-fact)))
                samples = sample(1:length(prob), 1000, TRUE, prob)
            } else {
                samples = samples[[2,i]]
            }
            fact = fact[samples]
            cfact = cfact[samples]
        }
        if (exp != BA_varname) {
            #mask = fact== 1 & cfact==1   
            fact = -log(1-fact*0.999999999)
            cfact = -log(1-cfact*0.999999999)
            #if (length(fact) != 1000) browser()
        }
         
        return(list(sort(cfact)/sort(fact), samples))

    }
    outs = mapply(for_gcm, gcms, 1:length(gcms))
    
    afs = as.vector(unlist(outs[1,]))
    afs = afs[afs != 1]
    
    plot_af(afs, xp/3 + xoffset, 
            name = paste(cfactual_name[2], min(years[[2]]), name), csv_out_name = 'AF-futr',
            col = col, bwidth = 0.025, ...)
    return(outs)
}

futr_rr_calc <- function(dir, region, factual_name, cfactual_name, exp, mnths, years,
                         BA, xp, xoffset, samples = NULL, name = name, col = col, width = 0.05, 
                         ylim0 = 0, yearss = NULL, ...) {
    
    
    futr_annotaion(exp, factual_name, xp, xoffset, width, ylim0, years, yearss)

    xs = xoffset + xp/3 + width*0.5*c(-1, 1)
    for_gcm <- function(i) {
        gcm = gcms[[i]]
        fact = openDat(dir, region, paste0(factual_name, '/', gcm), exp, cell_sample, mnths, years[[1]])
        cfact = openDat(dir, region,  paste0(cfactual_name, '/', gcm), exp, cell_sample, mnths, years[[2]])
        
        if (!background_BA && (is.null(samples) || exp == BA_varname)) {
             
            prob = exp(BA*log(fact) + (1.0-BA)*log((1-fact)))
            samples = sample(1:length(prob), 1000, TRUE, prob)
            
            facts = sort(fact[samples])
            cfacts = sort(cfact[samples])
            prob1 = cumm_pdf(facts, BA)
            prob2 = cumm_pdf(cfacts, BA)
            
            rr = prob2/prob1
            samples = list(prob1, prob2, samples)
            
        } else {    
            if (background_BA || is.null(samples)) { 
                rr = mean(cfact/fact)   
            } else {
                prob = samples[[2, i]]
                rr = mean(cfact[prob[[3]]]/fact[prob[[3]]])
            }
        }
        return(list(rr, samples))
    }
    outs = sapply(1:length(gcms), for_gcm)
    rr = as.vector(unlist(outs[1,]))
    
    plot_af(as.vector(rr), xp/3 + xoffset, 
            name = paste(cfactual_name[2], min(years[[2]]), name), csv_out_name = 'RR-futr',
            col = col, bwidth = 0.025, ...)
    
    return(outs)
}

plot_region <- function(region, HadGEM_dir, ISIMIP_dir, mnths, years,
                        empty_plot = new_empty_plot_logit, 
                        att_FUN = att_af_calc, futr_FUN = futr_af_calc,
                        ylim = list(NULL, NULL, NULL), mitigate = False, 
                        reduced = TRUE, csv_out = csv_out) {
    
    add_run <- function(dir, factual_name = "factual-", cfactual_name = "counterfactual-",
                        xoffset = 0.0, years = 2025, mnths = NULL, 
                        plot_FUN = att_FUN, BA = 0.0, ylim0 = 0.0, ...) {
            
        
        add_experiemtnt <- function(exp = BA_varname, col = 'red', xp = 0.25, 
                                    name = 'Burned Areas',
                                    samples = NULL) {
            
            outs = plot_FUN(dir, region, factual_name, cfactual_name, exp, mnths, years,
                        BA, xp, xoffset, samples, name = name, col = col, ylim0 = ylim0, 
                        csv_out = csv_out, ...)
            return(outs)
        } 
        outs_BA = add_experiemtnt(col = cols[1])
        exps = c("standard-Fuel", "standard-Moisture")#, "standard-Ignition","standard-Suppression")
        outs_contol = mapply(add_experiemtnt, exps, cols[2:3], c(0.5, 0.75), 
                             c("Fuel connectivity", "Dryness"),
                            MoreArgs = list(samples = outs_BA))
        
        return(c(outs_BA, outs_contol))#samples[[2]])
    }
    
    ylim0 = empty_plot(xlim = c(0, 1), ylim = ylim[[1]])
    if (!background_BA) {
        BA = add_run(HadGEM_dir, mnths = mnths, BA = NULL, ylim0 = ylim0)[[2]]
        text(x = 0.5, y = ylim0, adj = c(0.5, 1.3), font = 2, 'HadGEM3-A', xpd = NA)
    } else {
        BA = add_run(ISIMIP_dir, years =  2002:2019, BA = NULL, ylim0 = ylim0) 
        text(x = 0.5, y = ylim0, adj = c(0.5, 1.3), font = 2, 'ISIMIP3a', xpd = NA) 

    }
    #if (reduced) hadgemtxt = 'HadGEM3-A'
    #    else hadgemtxt = 'HadGEM3-A full ensemble'
    #text(x = 0.5, y = ylim0, adj = c(0.5, 1.3), font = 2, hadgemtxt, xpd = NA)

    #if (!reduced) {
    #    add_run(HadGEM_dir, mnths = mnths, cfactual_name = "counterfactual_mean-", 
    #            xoffset = 1, BA = BA, ylim0 = ylim0)
    #    add_run(ISIMIP_dir, xoffset = 2, years =  2002:2019, BA = BA, ylim0 = ylim0)        
    #    text(x = 1.5, y = ylim0, adj = c(0.5, 0.3), font = 2, 'HadGEM3-A ensemble mean', xpd = NA)
    #    text(x = 2.5, y = ylim0, adj = c(0.5, 0.3), font = 2, 'ISIMIP3a', xpd = NA)
    #}
    plot.new()

    if (reduced) {
        yearss = list(2030:2039, 2040:2049, 2090:2099)
    } else {
        yearss = list(2020:2029, 2030:2039, 2040:2049, 2050:2059, 2060:2069, 
                      2070:2079, 2080:2089, 2090:2099)
    }
    ylim0 = empty_plot(xlim = c(0, length(yearss)-0.075), ylab = '', xaxs = 'i', ylim = ylim[[2]])
    #axis(1)
    for_ssp <- function(ssp, xmini_off, yearss) {
        subdir = c("historical", ssp)
        
        for_yrss <- function(yrss, xoffset) {
            
            add_run(ISIMIP_dir, xoffset = xoffset-1 + xmini_off, years =  list(2010:2019,yrss), 
                    mnths = mnths,yearss = yearss,
                    plot_FUN = futr_FUN, factual_name = subdir, cfactual_name = subdir, 
                    BA = BA, ylim0 = ylim0)
        }
        mapply(for_yrss, yearss, 1:length(yearss), SIMPLIFY = FALSE)
    }
    outs = mapply(for_ssp, c("ssp126", "ssp370", "ssp585"), c(0, 0.3, 0.6), 
                  MoreArgs = list(yearss), SIMPLIFY = FALSE)

    plot_ssp_diff <- function(i, ssp, xmini_off, years, ..., ylim0 = 1) {
        ssp1 = outs[[i+1]]
        ssp2 = outs[[i]]
        for_time <- function(yrss, xoffset) {
            ssp1 = ssp1[[xoffset]]
            ssp1 = ssp1[seq(1, length(ssp1), by = 2)]
            ssp2 = ssp2[[xoffset]]
            ssp2 = ssp2[seq(1, length(ssp2), by = 2)]
            for_control <- function(i, exp, name = '', xp = 0.25,...) {
                print("yyyaaayyy!!")
                print(ssp)
                 
                futr_annotaion(exp, ssp, xp, xmini_off + xoffset - 1,0.05, ylim0, 
                               list(yearss[[1]], yrss), yearss)
                ssp10 = ssp1; ssp20 = ssp2
                index = ((i-1)*5+1):(i*5)
                ssp1 = sort(unlist(ssp1[index]))
                ssp2 = sort(unlist(ssp2[index]))
                
                af = 100*(ssp2-ssp1)/(ssp1-1)
                
                plot_af(af, xmini_off + xp/3 + xoffset-1, name = name, 
                     csv_out = csv_out, csv_out_name = 'AF-migigation', bwidth = 0.025, 
                     FUN = af_mit_tscale, ...)
            }
            mapply(for_control, 1:3, c(BA_varname, "standard-Fuel", "standard-Moisture"), 
                  c('Burned Areas', "Fuel connectivity", "Dryness"),
                    col = cols, xp = c(0.25, 0.5, 0.75),
                    MoreArgs = list(...))
        }
        
        mapply(for_time, yearss, 1:length(yearss), SIMPLIFY = FALSE)
        
    }
    if (mitigate) {
        plot(c(0,1), c(0, 1), type = 'n', xaxt = 'n', yaxt = 'n', axes = FALSE)
        
        legend_point <- function(col, x, name) {
            plot_af(runif(1000, 0.8, 1), x, col, FUN = function(i) i)
            text(x, 0.8, adj = c(0.5, 1.1), name)
        }
        mapply(legend_point, cols, c(0.25, 0.5, 0.75), 
               c('Burned\nArea', 'Fuel\nLoad', 'Dryness'))
    
        plot.new()
        ylim0 = new_empty_plot_mitigate(xlim = c(0, length(yearss)-0.075), 
                           ylab = '', xaxs = 'i', ylim = ylim[[3]],
                           add_xlabs = TRUE)
        mapply(plot_ssp_diff, 1:2, c("Mitigation\npotential", "Already\nmitigated"), c(0, 0.3), 
                      MoreArgs = list(yearss, ylim0 = ylim0)) 
        
        mapply(function(x, yrss)
              futr_annotaion(BA_varname, "ssp585", 0.3, x-0.4,0.05, ylim0, 
                             list(yearss[[1]], yrss), yearss, addSSPlab = FALSE), 
                              1:length(yearss), yearss)   
    } 
}

plot_region_all_plots <- function(region, mnths, years, ylim1 = NULL, ylim2 = NULL, reduced = TRUE) {
    extra_filename = paste(cell_sample, c('', 'reduced')[reduced+1], 
                 c('event', 'background')[background_BA+1], region, sep = '-')
    csv_out = paste0("outputs/SoW_att_outlook", extra_filename, '.csv')
    file.create(csv_out)
    
    fout = paste("figs/att_outlook", extra_filename, '.png', sep = '-')
    
    #png(fout, width = 14 - 7*reduced, height = 7, units = 'in', res = 300)
    if (reduced)
        widths = c(0.15, 0.05, 0.4)
    else
        widths = c(0.09, 0.025, 0.46)
    layout(rbind(1:3, 4:6, 7:9), widths = widths)
    par(oma = c(2, 5, 2, 5), mar = c(0.5, 0, 0.5, 0))
        plot_region(region,  HadGEM_dir, ISIMIP_dir, mnths, years, 
                    reduced = reduced, 
                     ylim = ylim1, csv_out = csv_out, mitigate = TRUE)
        plot_region(region,  HadGEM_dir, ISIMIP_dir, mnths, years, 
                    empty_plot = new_empty_plot_rr,
                    att_FUN = att_rr_calc, futr_FUN = futr_rr_calc, ylim = ylim2, 
                    reduced = reduced, csv_out = csv_out)
    dev.off()
}


cols = c("#B50000", "#E98400", "#0096A1")#, "#EE0074")#, "purple", "grey")

HadGEM_dir = "outputs/outputs_scratch/SoW2526/attribution-HadGEM-test29-fuelcf4/<<region>>/time_series/_16-frac_points_0.5/"
ISIMIP_dir = "outputs/outputs_scratch/SoW2526/isimip/full-4-notree-notreechange/<<region>>/time_series/_15-frac_points_0.5/"
ISIMIP_dir = "outputs/outputs_scratch/SoW2526/isimip/full-5-notree-notreechange/<<region>>/time_series/_15-frac_points_0.5/"
gcms = c("GFDL-ESM4-", "IPSL-CM6A-LR-", "MPI-ESM1-2-HR-", "MRI-ESM2-0-", "UKESM1-0-LL-")

regions = c("Northwest Iberia", "Midwestern Canadian Shield forests", 
            "Chilean Temperate Forests and Matorral")
            #, "Scottish_Highlands", "Southeast_South_Korea")

ylim1 = list(NULL, NULL, NULL)
ylim2 = list(NULL, NULL, NULL)
ylim1 = list(list(c(0.48, 9E9), c(0.85, 4.2),c(-200, 200)),
             list(c(0.65, 9E9), c(0.65, 8.2), c(-200, 200)),
             list(c(0.5, 9E9), c(0.75, 3), c(-200, 2000)))

years = list(2025, 2025, 2026)#, 2025, 2025)
mnths = list(c('08'), c('07', '08'), c('01', '02', '03'))#, c('06', '07'), c('03'))#
cell_sample = "mean"
background_BA = FALSE
BA_varname = "Evaluate"

mapply(plot_region_all_plots,regions, mnths, years, ylim1, ylim2)
mapply(plot_region_all_plots,regions, mnths, years, ylim1, ylim2, reduced = FALSE)


cell_sample = "pc-95.0"
mapply(plot_region_all_plots,regions, mnths, years, ylim1, ylim2, reduced = FALSE)

mnths = c(paste0('0', 1:9), 10:12)
mnths = list(mnths, mnths, mnths)
cell_sample = "mean"
background_BA = TRUE

mapply(plot_region_all_plots,regions, mnths, years, ylim1, ylim2, reduced = FALSE)
