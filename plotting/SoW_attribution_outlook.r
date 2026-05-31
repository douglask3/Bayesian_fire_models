
openDat <- function(dir, subdir, experiment, area, mnths, years) {
    file = gsub('<<region>>', gsub(' ', '_', region), dir)
    file = paste(file, subdir, area, '/members/absolute/', 
                 paste0(experiment, '.csv'),
                 sep = '/')

    print(file)
    dat = read.csv(file)
       
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

af_tscale <- function(x) x/(1+x)
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
    text(x = xpos - 0.1, y = pcs[3], paste0(likelihood, '%'), adj = 1.1) 
    text(x = xpos, y = min(pcs), adj = 1.1, srt = 45, name)
}

regions = c("Midwestern Canadian Shield forests", 
            "Chilean Temperate Forests and Matorral", 
            "Northwest Iberia")

HadGEM_dir = "outputs/outputs_scratch/SoW2526/attribution-HadGEM-test29-fuelcf4/<<region>>/time_series/_16-frac_points_0.5/"
ISIMIP_dir = "outputs/outputs_scratch/SoW2526/isimip/full-3/<<region>>/time_series/_15-frac_points_0.5/"

region = tail(regions, 1)
mnths = c('01','02', '03', '04', '05', '06', '07','08', '09', '10', '11', '12')
mnths = c('09')
years = c(2025)


new_empty_plot_logit <- function(xlim = c(0, 2), ylab =  'Amplifcation factor') {
    plot(c(0, 2), c(0, 1), xlab = '', ylab = '', type = 'n', yaxt = 'n', yaxs = 'i')
    
    labels = c(0, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64, 128, 1000000)
    at = af_tscale(labels)
    labels = c('0', '', '', '', '', '', '1/4', '1/2', 'no\nchange', '2', '4', '', '', '', '', '', 'All from\nclimate')
    axis(2, at = at, labels = labels)
    mtext(side = 2, line = 2.75, ylab)
}
    
plot_region <- function(region, HadGEM_dir, ISIMIP_dir, mnths, years) {
    
    add_run <- function(dir, factual_name = "factual-", cfactual_name = "counterfactual-",
                        xoffset = 0.0, years = 2025, mnths = NULL) {
        add_experiemtnt <- function(exp = "Evaluate", col = 'red', xp = 0.25, 
                                    name = 'Burned Areas',
                                    samples = NULL) {
            
            fact = openDat(dir, factual_name, exp, "mean", mnths, years)
            cfact = openDat(dir, cfactual_name, exp, "mean", mnths, years)

            BA = 0.1
        
            if (is.null(samples)) {
                prob = exp(BA*log(fact) + (1.0-BA)*log((1-fact)))
                samples = sample(1:length(prob), 1000, TRUE, prob)
            }
            facts = fact[samples]
            cfacts = cfact[samples] 
            #browser()
            plot_af(facts, cfacts, xp + xoffset, name, col = col)
            return(samples)
        } 
        samples = add_experiemtnt(col = "#E98400")
        exps = c("standard-Fuel", "standard-Moisture")#, "standard-Ignition","standard-Suppression")
        cols = c("#0096A1", "#EE0074")#, "purple", "gret")
        mapply(add_experiemtnt, exps, cols, c(0.5, 0.75), c("Fuel connectivity", "Dryness"),
               MoreArgs = list(samples = samples))
    }
    par(mfrow = c(2, 2))
    new_empty_plot_logit()
    add_run(HadGEM_dir, mnths = mnths)
    add_run(ISIMIP_dir, xoffset = 1, years = 2005:2014)
    text(x = 0.5, y = 0, adj = c(0.5, -0.3), font = 2, 'HadGEM3-A')
    text(x = 1.5, y = 0, adj = c(0.5, -0.3), font = 2, 'ISIMIP3a')

    new_empty_plot_logit(xlim = c(0, 9))
    browser()
}
plot_region(region,  HadGEM_dir, ISIMIP_dir, mnths, years)
