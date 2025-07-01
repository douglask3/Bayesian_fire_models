
library(magick)


logit <- function(x)  log(x/(1-x))
logistic <- function(y) 1/(1+exp(-y))

add_polygon <- function(y, x, col, name, mu = NULL) {
    polygon(x -0.3*c(-1, 1, 1, -1, -1), c(0, 0, 1, 1, 0),
            border = col, col = paste0(col, '33'))
    polygon(x -0.3*c(-1, 1, 1, -1, -1), y*c(0, 0, 1, 1, 0), 
            border = NA, col = paste0(col, '88'))
    polygon(x -0.3*c(-1, 1, 1, -1, -1), c(0, 0, 1, 1, 0), border = col)
    if (!is.null(mu)) lines(x + c(-0.3, 0.3), c(mu, mu), col = col)
    text(x = x, y = 0, adj = c(0, 1), name, srt = -45, xpd = NA)
}

run_control <- function(id, control, ..., load = FALSE, typical = FALSE) {
    mu = control[[id]][1]
    sigma = control[[id]][2]
    y = logistic(rnorm(1, logit(mu), sigma))
    if (load) y = 0.9 + 0.1 * y
    if (typical) y = mu
    add_polygon(y, col = control[[3]], mu = mu,...)
    return(y)
}

controls = list("Fuel" = list(c(0.7, 0.2), c(0.6, 0.4), "#0096a1"), 
                "Moisture" = list(c(0.5, 0.4), c(0.65, 0.9), "#7a44ff"), 
                Ignitions = list(c(0.4, 0.6), c(0.4, 0.9), "#b50000"), 
                Humans = list(c(0.7, 0.05), c(0.7, 0.1), "#ee007f"),
                Others = list(c(0.9, 0.1), c(0.9, 0.1), "#999999"))

Factor = c("Fire" =  "#e98400")

for_instance <- function(id = 1, ...) {
    outs = mapply(run_control, id, controls, 1:length(controls), name = names(controls), ...)
    fire = prod(outs)#^(1/4)
    add_polygon(fire, length(controls) + 2, Factor, names(Factor))
    return(fire)
}

for_climate <- function(id, fires, ...) {
    plot(c(0, 7.2), c(-0.2, 1.2), type = 'n', axes = FALSE, xlab = '', ylab = '')
    fires = c(fires, 100*for_instance(id, ...))
    fires[fires > 40] = 40
    hist(fires, xlab = 'Burned Area (%)', xlim = c(0, 40),
         ylab = '', yaxt = 'n', col = Factor, main = '', 
         breaks = seq(0, 42, 2))#breaks = round(sqrt(length(fires))))
    
    return(fires)
}

run <- function() {
    fires1 = fires2 = c()
    plots <- list()
    for (i in 1:500) {
        fname = paste0("docs/figs/ConFLAME_eg/frame-", i, '.png')
        png(fname, height = 4, width = 7.2, res = 100, units = 'in')
        par(mfrow = c(2, 2), mar = rep(1, 4), oma = c(2, 0, 0, 0))
        
        fires1 = for_climate(1, fires1)
        fires2 = for_climate(2, fires2)
        dev.off()
        plots[[i]] <- image_read(fname)
    }
    gif <- image_animate(image_join(plots), fps = 5)    
    image_write(gif, "docs/figs/ConFLAME_eg.gif")
}

fname = paste0("docs/figs/ConFLAME_eg_exteme.png") 
png(fname, height = 4, width = 7.2, res = 100, units = 'in')
par(mfrow = c(2, 2), mar = rep(1, 4), oma = c(2, 0, 0, 0))
for_climate(1, c(), load = TRUE)
dev.off()

fname = paste0("docs/figs/ConFLAME_eg_normal.png")
png(fname, height = 4, width = 7.2, res = 100, units = 'in')
par(mfrow = c(2, 2), mar = rep(1, 4), oma = c(2, 0, 0, 0))
for_climate(1, c(), typical = TRUE)
dev.off()

   
run() 

    
    
    
