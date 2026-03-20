legendColBar <- function(xx = c(0.3, 0.7), yy = c(0.1, 0.9), cols, limits, switch = FALSE,
                             extend_max = TRUE, extend_min = TRUE, minLab = '', maxLab = '',
                             units = '', transpose = FALSE, oneSideLabels = TRUE, adj = 0.5,
                             xtext_pos_scale = 0.3, tagTicks = TRUE,
                             add = FALSE) {
        if (!add) plot(c(0, 1), c(0, 1), axes = FALSE, type = 'n')
        
        if (length(cols) != (length(limits) + 1))
            cols = make_col_vector(cols, ncols = length(limits)+1)
        ys = seq(yy[1], yy[2], length.out = length(cols) +1)
        
        addBox <- function(y1, y2, yi, col, ...) {
            print(yi)
            polyFun <- function(x, y, ...) {
                if (transpose) {xi = x; x = y; y = xi}
                    polygon(x, y, col = col, lwd = 2, xpd = NA, ...)
            }
            F1 <- function() polyFun(c(xx, mean(xx), xx[1]), c(y2, y2, y1, y2), ...)
            F2 <- function() polyFun(c(xx, mean(xx), xx[1]), c(y1, y1, y2, y1), ...)
            if (yi == 1 && extend_min) 
                if (switch) F1() else F2()
            else if (yi == length(cols) && extend_max) 
                if (switch) F2() else F1()
            else polyFun(c(xx, rev(xx), xx[1]), c(y2, y2, y1, y1, y2), ...)
                 
        }
        id = 1:length(cols)
        if (switch) {
            id = rev(id)
            cols = rev(cols)
            limits = rev(limits)
        }
        mapply(addBox, ys[-1], head(ys, -1), id, cols)
        if (length(limits) > 15)  mapply(addBox, ys[-1], head(ys, -1), id, cols, border = NA)
        
        y = ys
        
        if (!extend_min && minLab == "" && limits[1] == 0) y[2] = mean(y[1:2])
        limits = c(minLab, limits, maxLab)
        limits[(abs(as.numeric(limits)) < 0.00000000001)] = '0'
        if ((nchar(units) == 1) && tagTicks)
            limits[limits != ''] = paste0(limits[limits != ''], units)
        else
            limits[tail(which(limits != ''), 1)] = paste0(limits[tail(which(limits != ''), 1)], units)
     
        x = rep(xx[2] + diff(xx) *xtext_pos_scale, length(y))
        if (is.na(oneSideLabels)) x = rep(xx[1] - diff(xx) *xtext_pos_scale, length(y))
        else if (!oneSideLabels) x[seq(1, length(x), by = 2)] = xx[1] - diff(xx) *xtext_pos_scale
        
        if (transpose) { xi = x; x = y; y = xi}
        
        if (length(x) > 25) {
            index = round(seq(1, length(x), length.out = 11))
            x = x[index]
            y = y[index]
            limits = limits[index]
        }
        mapply( text.units,x, y, limits, xpd = NA, adj = adj)
        
    }
