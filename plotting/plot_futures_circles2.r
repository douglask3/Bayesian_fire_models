
plot_nested_circles <- function(diameters, theta = 135, col_fill = '#00000002',
                                border_col = 'black', name = '', clip = 0.99) {

    rs = diameters / 2
    a = seq(0, 2 * pi, length.out = 500)
    if (is.na(theta)) {
        angle = 0
        xshift = 0
        yshift = 0
    } else {
        angle = theta * pi / 180
        xshift = cos(angle)
        yshift = sin(angle)
    }
    # Angle in radians
    for_circle <- function(r, border = NA, col = col_fill, ...) {
        polygon(r * cos(a) + xshift*r, r * sin(a) + yshift*r,
          border = border, col =col, xpd = NA,...)
    }
    clipv = quantile(rs, clip)
    lapply(sample(rs[rs<=clipv], min(length(rs), 100)), for_circle)
     
    for (i in 1:3) for_circle(clipv)
    ranges = quantile(rs, c(0.1, 0.5, 0.90))
    for_circle(ranges[1], '#FFFFFF66', '#00000000')
    for_circle(ranges[1], border_col, '#00000000', lty = 2, lwd = 2)
    for_circle(ranges[2], border_col, '#00000000', lty = 1, lwd = 2)
    for_circle(ranges[3], '#FFFFFF66', '#00000000')
    for_circle(ranges[3], border_col, '#00000000', lty = 2, lwd = 2)
    text(x = (0.076+ranges[3]) * (cos(angle+0.52*-1) + xshift), 
         y = (0.076+ranges[3]) * (sin(angle+0.52*-1) + yshift), 
         name, font = 2, col = border_col, xpd = NA)
    return()
}

variable = "Burned_Area"
yearss = c("2030_2039", "2040_2049", "2090_2099")
regions = c("Northwest_Iberia", "Midwestern_Canadian_Shield_forests", "Chilean_Temperate_Forests_and_Matorral")
type = "mean--normal"
dir = "outputs/outputs_scratch/SoW2526/Zenodo/<<region>>//isimip/time_series/_15-frac_points_0.5/futureAFs/"

ssps = c("ssp126", "ssp370", "ssp585")

load_and_plot <- function(years, tplot = TRUE, names = rep('', 3), region = regions[1], lines = -2, obs_txt = '', transpose = FALSE, clip = 0.99,...) {
    dir = gsub('<<region>>', region, dir)
    files = list.files(dir)
    #browser()
    for (test in c(variable, years, type)) 
        files = files[grepl(test, files)]

    load_ssp <- function(ssp) {
        load(paste0(dir, files[grep(ssp, files)]))
        if (transpose) outs = t(outs)
        out = unlist(outs[1,])
        #if (ssp == "ssp370")
        #    browser() 
        #out = out[(out<0.999) | (out>1.001)]
        out = out[outs<1000]
        #browser()
        return((out)^0.5)
    }
    dats = lapply(ssps, load_ssp)
    
    mx = quantile(unlist(dats), clip)
    mx = min(mx, 10)
    dats = lapply(dats, function(dat) dat/mx)
    if (!tplot) return(mx)
    plot(NA,
       xlim = c(-1, 1),
       ylim = c(-1, 1),
       asp = 1,
       axes = FALSE,
       xlab = "", ylab = "",
       frame.plot = FALSE)
    
    #plot_nested_circles(rev(sort(dats[[1]])), 45)
    plot_nested_circles(1/mx, NaN, 'black')
    #browser()
    plot_nested_circles(rev(sort(dats[[3]])), 180, '#C7403D05', '#C7403D', names[3], 
                        clip = clip, ...)
    plot_nested_circles(rev(sort(dats[[2]])), 60, '#E2722605', '#E27226', names[2], 
                        clip = clip, ...)
    plot_nested_circles(rev(sort(dats[[1]])), 300, '#00878705', '#008787', names[1], 
                        clip = clip, ...)
    plot_nested_circles(1/mx, NaN, '#00000000', 'white', ...)
    if (names[1] != '') 
        text(obs_txt, x=0, y=0, font = 2, col = 'white')
    mtext(side = 1, adj = 0.5, gsub('_', ' to\n', years), line = lines)
}

plot_region <- function(region, lines = -2, ...) {
    out = sapply(yearss, load_and_plot, tplot = FALSE, region = region, ...)
    #cairo_pdf(paste0('figs/future_circles', region, '.pdf'), 
    #    height = max(out)*7.2/sum(out), width = 7.2, family = "Work Sans")#, units = 'in', res = 300)
    png(paste0('figs/future_circles', region, '.png'), 
        height = max(out)*7.2/sum(out), width = 7.2, units = 'in', res = 300)
    layout(rbind(1:3, 1:3), widths = out)
    par(mar = rep(0.0, 4), family = "Work Sans")
    mapply(load_and_plot, yearss, T, list(rep('', 3), rep('', 3), ssps), 
           region = region, lines = lines, ...)#, mx  = max(out))
    title(gsub("_", " ", region), outer = T, line = lines)
    dev.off()
}
#mapply(plot_region, regions, c(-6, -2, -4), obs_txt = c('Aug\n2025', 'Jul & Aug\n 2025', 'Jan & Feb\n2026'))


variable = "Evaluate"
yearss = c("2030_2039", "2040_2049", "2090_2099")


regions = c("UK")
type = "allModels-mean"
dir = "outputs/outputs_scratch/UK/isimip/final_run-<<region>>/data_store/futures/"
mapply(plot_region, regions, c(-10), obs_txt = c(''), transpose = TRUE)


regions = c("England")
type = "allModels-mean"
dir = "outputs/outputs_scratch/UK/isimip/final_run-<<region>>/data_store/futures/"
mapply(plot_region, regions, c(-4), obs_txt = c(''), transpose = TRUE, clip = 0.95)


regions = c("Scotland")
type = "allModels-mean"
dir = "outputs/outputs_scratch/UK/isimip/final_run-<<region>>/data_store/futures/"
mapply(plot_region, regions, c(-4), obs_txt = c(''), transpose = TRUE, clip = 0.95)


regions = c("Wales")
type = "allModels-mean"
dir = "outputs/outputs_scratch/UK/isimip/final_run-<<region>>/data_store/futures/"
mapply(plot_region, regions, c(-4), obs_txt = c(''), transpose = TRUE, clip = 0.95)


