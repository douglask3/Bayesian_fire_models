graphics.off()


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

log10_plus <- function(x) (log10(x) + 1)/2

af_iscale <- function(x) x/(1-x)


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
    
     if (!add_xlabs) labels_txt[] = ''
    axis(2, at = at, labels = labels_txt)

    mtext(side = 2, line = 3, ylab)
    for (y in at)
        lines(c(-9E9, 9E9), c(y, y), col= 'grey', lty = 2)
    return(ylim[1])
}

new_empty_plot_rr <- function(...) {

    labels = c(1/10, 1/5, 1/2, 1, 2, 5, 8, 10)
    labels_txt = c('1/10', '1/5', '1/2', '1', '2', '5', '8', '10')
    new_empty_plot(log10_plus, labels, labels_txt, ...)
}

new_empty_plot_logit <- function(...) {
    labels = c(0, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 2/3, 1, 
               1.5, 2, 4, 8, 16, 32, 64, 128, 1000000)
    labels_txt = c('0', '', '', '', '', '', '1/4', '1/2', '2/3', 'no\nchange', '1 1/2',
                   '2', '4', '', '', '', '', '', 'All from\nclimate')
    out = new_empty_plot(af_tscale, labels, labels_txt,...)
    
    #if (ylab == '') {   
    #    mtext(side =4, line = 3, "% explained by climate change")
    #    axis(4, seq(0, 1, length.out=  9), seq(-100, 100,length.out=  9))
    #}
    return(out)
}

##########################################################
## plotting functions                                   ##
##########################################################

plot_af <- function(af, xpos = 1, col = 'red', name = '', bar = TRUE, label = '', 
                     csv_out = NULL, csv_out_name = 'AF', bwidth = 0.1, ...) {
    bwidth_bar = 0.05*bwidth^(0.33)/0.1^(0.33)
    
    if (bar) {
        pc = quantile(af, c(0.05, 0.25, 0.5, 0.75, 0.95), na.rm = TRUE)
        outline = af_tscale(range(af, na.rm = TRUE))
        
        print(pc-1)
        pcs = af_tscale(pc)
        for (i in c(1, 5))
            lines(xpos + c(-bwidth_bar, bwidth_bar), rep(pcs[i], 2), col = col)
        lines(rep(xpos, 2), pcs[c(1, 5)], col = col)
        polygon(xpos + bwidth*c(-1, -1, 1, 1), pcs[c(2, 4, 4, 2)], border = NA, col = col) 
        lines(xpos + bwidth*c(-1, 1), rep(pcs[3], 2), lwd = 2, xpd = NA)        
    }
    likelihood = round(mean(af>1)*100 + mean(af==1)*50)
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
            BA = openDat(dir, region, factual_name, "observation", cell_sample, mnths, years)
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
        if (is.null(BA))
            BA = openDat(dir, region, factual_name, "observation", cell_sample, mnths, years)
        prob1 = cumm_pdf(fact, BA)
        prob2 = cumm_pdf(cfact, BA)
        
        rr = prob1/prob2
        samples = list(list(prob1, prob2), BA)
    } else {
        BA = samples[[2]]
        samples = samples[[1]]
        rr = (fact* samples[[1]])/(cfact * samples[[2]])
    }
    
    plot_af(rr, xp + xoffset, csv_out_name = 'rr', ...)
    
    
    return(samples)
}


futr_annotaion <- function(exp, factual_name, xp, xoffset, width, ylim0, years, yearss)  {
    if (exp == BA_varname) {
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
 
    if (years[[2]][1]== yearss[[1]][1] && exp == BA_varname) 
        text(xoffset + xp/2, ylim0, adj = c(-1, 0.5), srt = 90, factual_name[2])
}

futr_af_calc <- function(dir, region, factual_name, cfactual_name, exp, mnths, years,
                         BA, xp, xoffset, samples = NULL, name = name, col = col, width = 0.05, 
                        ylim0 = 0, yearss = NULL, ...) {
    
    futr_annotaion(exp, factual_name, xp, xoffset,width, ylim0, years, yearss)
    
    xs = xoffset + xp/3 + width*0.5*c(-1, 1)
    for_gcm <- function(gcm) {
        fact = openDat(dir, region, paste0(factual_name, '/', gcm), exp, cell_sample, mnths, years[[1]])
        cfact = openDat(dir, region,  paste0(cfactual_name, '/', gcm), exp, cell_sample, mnths, years[[2]])
        
        if (!background_BA)  {
            prob = exp(BA*log(fact) + (1.0-BA)*log((1-fact)))
            samples = sample(1:length(prob), 1000, TRUE, prob)
            
            fact = fact[samples]
            cfact = cfact[samples]
        }
        if (exp != BA_varname) {
            mask = fact== 1 & cfact==1   
            fact = -log(1-fact[!mask]*0.999999999)
            cfact = -log(1-cfact[!mask]*0.999999999)
        }
        return(sort(cfact)/sort(fact))

    }
    afs = sapply(gcms, for_gcm)
    afs = as.vector(unlist(afs))
    
    plot_af(afs, xp/3 + xoffset, 
            name = paste(cfactual_name[2], min(years[[2]]), name), csv_out_name = 'AF-futr',
            col = col, bwidth = 0.025, ...)
        
    
    #pc =  quantile(afs, c(0.05, 0.25, 0.5, 0.75, 0.95), na.rm = TRUE)
    #out = cbind(paste(cfactual_name[2], min(years[[2]]), name), 'AFs', names(pc), round(pc, 2))
    ##liki = (mean(pc>1) + 0.5 * mean(pc==1))*100
    #if (is.na(liki)) 
    #    browser()
    #out = rbind(out, c(name, 'likelihood', '%', liki))
#
    #if (!is.null(csv_out)) 
     #       write.table(out, file = csv_out, sep = ",", 
      #                  append = TRUE, col.names = FALSE, row.names = FALSE)
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
        
        if (is.null(samples) || exp == BA_varname) {

            
            prob = exp(BA*log(fact) + (1.0-BA)*log((1-fact)))
            samples = sample(1:length(prob), 1000, TRUE, prob)
            
            facts = sort(fact[samples])
            cfacts = sort(cfact[samples])
            prob1 = cumm_pdf(facts, BA)
            prob2 = cumm_pdf(cfacts, BA)
            
            rr = prob2/prob1
            #rr = mean(cfact>BA)/mean(fact>BA)
            samples = list(prob1, prob2, samples)
            
            
        } else {        
            prob = samples[[i]]
            rr = mean(cfact[prob[[3]]]/fact[prob[[3]]])
        }
        return(list(rr, samples))
    }
    outs = sapply(1:length(gcms), for_gcm)
    rr = as.vector(unlist(outs[1,]))
    
    plot_af(as.vector(rr), xp/3 + xoffset, 
            name = paste(cfactual_name[2], min(years[[2]]), name), csv_out_name = 'RR-futr',
            col = col, bwidth = 0.025, ...)
    
    #pc =  quantile(rr, c(0.05, 0.25, 0.5, 0.75, 0.95), na.rm = TRUE)
    #out = cbind(paste(cfactual_name[2], min(years[[2]]), name), 'RR', names(pc), round(pc, 2))
    #out = rbind(out, c(name, 'likelihood', '%', (mean(pc>1) + 0.5 * mean(pc==1))*100))#
#
#    if (!is.null(csv_out)) 
#            write.table(out, file = csv_out, sep = ",", 
#                        append = TRUE, col.names = FALSE, row.names = FALSE)
    
    return(samples)
}

plot_region <- function(region, HadGEM_dir, ISIMIP_dir, mnths, years,
                        empty_plot = new_empty_plot_logit, 
                        att_FUN = att_af_calc, futr_FUN = futr_af_calc,
                        ylim = NULL, reduced = TRUE, csv_out = csv_out) {
    
    add_run <- function(dir, factual_name = "factual-", cfactual_name = "counterfactual-",
                        xoffset = 0.0, years = 2025, mnths = NULL, 
                        plot_FUN = att_FUN, BA = 0.0, ylim0 = 0.0, ...) {
            
        add_experiemtnt <- function(exp = BA_varname, col = 'red', xp = 0.25, 
                                    name = 'Burned Areas',
                                    samples = NULL) {
            
            samples = plot_FUN(dir, region, factual_name, cfactual_name, exp, mnths, years,
                        BA, xp, xoffset, samples, name = name, col = col, ylim0 = ylim0, 
                        csv_out = csv_out, ...)
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
    extra_filename = paste(cell_sample, c('', 'reduced')[reduced+1], 
                 c('event', 'background')[background_BA+1], region, sep = '-')
    csv_out = paste0("outputs/SoW_att_outlook", extra_filename, '.csv')
    file.create(csv_out)
    
    fout = paste("figs/att_outlook", extra_filename, '.png', sep = '-')
    
    png(fout, width = 14 - 7*reduced, height = 9, units = 'in', res = 300)
    if (reduced)
        widths = c(0.2, 0.02, 0.4)
    else
        widths = c(0.1, 0.02, 0.4)
    layout(rbind(1:3, 4:6), widths = c(0.2, 0.02, 0.4))
    par(oma = c(2, 5, 2, 5), mar = c(1, 0, 1, 0))
        plot_region(region,  HadGEM_dir, ISIMIP_dir, mnths, years, 
                    reduced = reduced, 
                     ylim = ylim1, csv_out = csv_out)
        plot_region(region,  HadGEM_dir, ISIMIP_dir, mnths, years, 
                    empty_plot = new_empty_plot_rr,
                    att_FUN = att_rr_calc, futr_FUN = futr_rr_calc, ylim = ylim2, 
                    reduced = reduced, csv_out = csv_out)
    dev.off()
}

HadGEM_dir = "outputs/outputs_scratch/SoW2526/attribution-HadGEM-test29-fuelcf4/<<region>>/time_series/_16-frac_points_0.5/"
ISIMIP_dir = "outputs/outputs_scratch/SoW2526/isimip/full-4-notree-notreechange/<<region>>/time_series/_15-frac_points_0.5/"

gcms = c("GFDL-ESM4-", "IPSL-CM6A-LR-", "MPI-ESM1-2-HR-", "MRI-ESM2-0-", "UKESM1-0-LL-")

regions = c("Northwest Iberia", "Midwestern Canadian Shield forests", 
            "Chilean Temperate Forests and Matorral")
            #, "Scottish_Highlands", "Southeast_South_Korea")

ylim1 = list(NULL, NULL, NULL)
ylim2 = list(NULL, NULL, NULL)
ylim1 = list(NULL, NULL, NULL)

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
