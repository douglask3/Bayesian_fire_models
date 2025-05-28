find_levels <- function(z, percentiles = seq(20, 80, by = 20), not0 = FALSE) {
    if (is.raster(z)) z = as.numeric(z[])[!is.na(as.numeric(z[]))]
    z = z[!is.na(z)]
    if (any(z < 0)) symetry = T else symetry = F
    if (not0) z = z[z!=0]
    # Find the corresponding quantiles
    quantiles = quantile(abs(z), probs = percentiles / 100)
    
    scaling_factor = 10^(ceiling(log10((quantiles))))  # Adjust based on the magnitude of your data

    # Round the scaled quantiles to the nearest nice looking numbers
    levels = round(quantiles / scaling_factor, 2) * scaling_factor
    levels = unique(levels)
    if (symetry) levels = c(-rev(levels), levels)
    
    return(levels)
}

find_levels_n <- function(z, nlvls = 9, not0, zeroAt0 = TRUE) {
   
    if (is.raster(z)) z = as.numeric(z[])[!is.na(as.numeric(z[]))]
    z = z[!is.na(z)]
    if (any(z < 0) && zeroAt0) {
        z = abs(z)
        symetry = T 
    } else symetry = F
    if (not0) z = z[z!=0]
    z = unique(z)
    i = 0
    levels = c()
    sigfig = 0

    if (symetry) nlvls = ceiling(nlvls/2)
     
    quants = head(seq(0, 1, length.out = nlvls + 2)[-1], -1)
    while (i < 20 && length(levels) < nlvls) {
        sigfig = sigfig + 1
        levels = quantile(z, quants, na.rm = TRUE)
        scaling_factor = 10^(ceiling(log10((levels))))
        
        levels = round(levels / scaling_factor, sigfig) * scaling_factor
        levels = unique(levels)
        print(levels)
        i = i + 1
    }
    levels = unique(signif(levels, 2))
    levels1 =  unique(signif(levels, 1))
    if (length(levels1) == length(levels)) levels = levels1
    if (symetry) levels = c(-rev(levels), levels)
    print(levels)
    
    return(levels)
}
