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



futr_af_calc <- function(dir, region, factual_name, cfactual_name, mnths, years,
                         BA, ...) {
    BA = openDat(HadGEM_dir, region, "factual-","observation", 
                             cell_sample, mnths, list(2025))

    for_gcm <- function(gcm, i) {
        fact = openDat(dir, region, paste0(factual_name, '/', gcm), 
                       experiment, cell_sample, mnths, years[[1]])
        cfact = openDat(dir, region,  paste0(cfactual_name, '/', gcm),
                        experiment, cell_sample, mnths, years[[2]])

        moist = openDat(dir, region, paste0(factual_name, '/', gcm), 
                      "standard-Moisture",cell_sample, mnths, years[[1]])
        
        
        #if (background_BA)  
        #    prob = exp(0*log(fact) + (1.0-0)*log((1-fact)))
        #else
        prob = exp(BA*log(fact) + (1.0-BA)*log((1-fact)))
        
        prob[moist>0.9] = 0.0
        samples = sample(1:length(prob), 1000, TRUE, prob)

        fact = fact[samples]
        cfact = cfact[samples]
         
        return(list(sort(cfact)/sort(fact), samples))

    }
    outs = mapply(for_gcm, gcms, 1:length(gcms))
    
    afs = as.vector(unlist(outs[1,]))
    afs = afs[afs != 1]

    return(afs)
}


ISIMIP_dir = "outputs/outputs_scratch/SoW2526/Zenodo/<<region>>/isimip/time_series/_15-frac_points_0.5/"
gcms = c("GFDL-ESM4-", "IPSL-CM6A-LR-", "MPI-ESM1-2-HR-", "MRI-ESM2-0-", "UKESM1-0-LL-")

regions = c("Northwest Iberia", "Midwestern Canadian Shield forests", 
            "Chilean Temperate Forests and Matorral", "Scottish Highlands")

HadGEM_dir = "outputs/outputs_scratch/SoW2526/Final2/<<region>>/hadgem3/time_series/_16-frac_points_1e-08/"

experiment = "Evaluate"
mnths = list(c('08'), c('07', '08'), c('01', '02', '03'), c('06', '07'))#, c('06', '07'), c('03'))
yearss = list(2030:2039, 2040:2049, 2090:2099)
cell_sample = "mean"

ssps = c("ssp126", "ssp370", "ssp585")

for_yrss <- function(yrss) {
    for_ssp <- function(ssp) {
        subdir = c("historical", ssp)
        outs = futr_af_calc(ISIMIP_dir, regions[1], years =  list(2010:2019, yrss), 
                mnths = mnths[[1]],
                factual_name = subdir, cfactual_name = subdir, 
                    BA = BA)

    }
    outs = lapply(ssps, for_ssp)
    browser()
}
outs = lapply(yearss, for_yrss)
