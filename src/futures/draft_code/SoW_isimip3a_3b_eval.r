

reds9 = c("#FFFBF7", "#F7EBDE", "#EFDBC6", "#E1CA9E", 
		  "#D6AE6B", "#C69242", "#2B7151", "#9C5108",
		  "#6B3008")
 
historical =  read.csv("outputs/outputs_scratch/SoW2526/isimip/full-5-notree-notreechange/Northwest_Iberia/time_series/_15-frac_points_0.5/factual-/mean/members/absolute/Evaluate.csv")

futures = read.csv("outputs/outputs_scratch/SoW2526/isimip/full-5-notree-notreechange/Northwest_Iberia/time_series/_15-frac_points_0.5/historical/UKESM1-0-LL-/mean/members/absolute/Evaluate.csv")
ix <- which(colnames(futures) %in% colnames(historical))
iy <- match(colnames(futures)[ix], colnames(historical))
futures = futures[, ix]
historical = historical[, iy]

mnths = c('08', '09')

h_years = 2010:2019
f_years = 2090:2099
h_years = f_years = 2000:2014
 
BA = 3
 
trans <- function (x)  {
	x = log10(x + 0.01)
	return(x)
}
inverse <- function(y) {
	y = 10^(y) - 0.01
	return(y)
 }
 
extract_dat <- function(yrs, dat) {
	which_yr_mn <- function(mnth, yr)  {
		which((substr(colnames(dat), 7, 8) == mnth) & 
                      (substr(colnames(dat), 2, 5) == yr))
	}
	test = sapply(yrs, function(yr) sapply(mnths, which_yr_mn, yr))
	(trans(100*unlist(dat[,test])/12))
}
 
hdat = extract_dat(h_years, historical)
fdat = extract_dat(f_years, futures)

mask = (hdat > trans(0.01)) & (fdat > trans(0.01)) & (hdat < trans(8)) & (fdat < trans(8))

hdat=hdat[mask];  fdat = fdat[mask]
 
xh =  seq(min(fdat, hdat), max(fdat, hdat), length.out = 101)

get_smoothed <- function(dat, dat0 = NULL) {

	h =  hist((dat), plot = FALSE, breaks = (xh))

	if (!is.null(dat0)) {

		h0 =  hist((dat0), plot = FALSE, breaks = (xh))

		out = smooth.spline((h$mids), h$density-h0$density, df = 10)

	} else {

		out = smooth.spline((h$mids), h$density, df = 10)#smooth.spline(x  = (h$mids), 

							#	y = cumsum(h$count), df = 10)

	}

	return(out)	

}
 
layout(rbind(1:2, 3:4), width = c(1, 1), height = c(1, 1))

par(mar = rep(1, 4), oma = c(3,3,3,3))
 
hh = get_smoothed(hdat)

fh = get_smoothed(fdat, hdat)
 
#hh$y =  c(hh$y[1],diff(hh$y))

fh$y =  hh$y + fh$y#c(fh$y[1],diff(fh$y))

#browser()	
 
plot(range(fh$y), range(fh$x),xlim = rev(range(fh$y)), type = 'n', axes = FALSE, xlab = '', ylab = '')

polygon(c(0, fh$y, 0), 

		c(min(fh$x), fh$x, max(fh$x)), col = '#FF000099', border = NA)

lines(c(-9E9, 9E9), rep(trans(BA), 2), lty = 2)	

#fdat = fdat - hdat

cols = densCols(hdat, fdat, colramp = colorRampPalette(blues9), bandwidth = 0.1)

plot(fdat~hdat, pch = 20, col = cols, cex = 2, axes = FALSE)

lines(c(-9E9, 9E9), c(-9E9, 9E9), lty = 2)

labels = c(0, 0.01, 0.1, 1,3)

at = trans(labels)
 
sapply(1:2, axis, labels = labels, at = at)

x = seq(quantile(xh, 0.00), quantile(xh, 1.0), length.out = 1000)

for (af in c(2, 5, 10, 20, 50, 100, 200)) {

	y = trans(inverse(x) * af)

	lines(x, y, lty = 3, col = 'grey')

	text(x[which.min(abs(y-trans(10)))], trans(10), af, xpd = NA, adj = c(1.5, 0.5), cex = 0.8, col = 'grey')

	y = trans(inverse(x) / af)

	lines(x, y, lty = 3, col = 'grey')

	text(trans(10), max(y), paste0('1/', af), xpd = NA, adj = c(0.5, 1.5), cex = 0.8, col = 'grey')

}

lines(trans(c(BA, BA)), c(-9E9, 9E9), lty = 2)

lines(c(-9E9, 9E9), trans(c(BA, BA)), lty = 2)
 
af_at_x <- function(xi) 

	inverse(quantile(fdat, mean(hdat<xi)))/inverse(quantile(hdat, mean(hdat<xi)))

afs0 = sapply(x, af_at_x)^2

afs = smooth.spline(afs0, df = 20)$y

#afs = smooth.spline(afs/afs[1], df = sqrt(length(x)))$y

lines(x, trans(afs*inverse(x)), col = 'red')
 
rr_at_x <- function(xi) 

	mean(fdat>xi)/mean(hdat>xi)

rrs = sapply(x, rr_at_x)
 
mx = max(fh$y, hh$y)

fh$y = (fh$y/mx) * diff(range(afs))+min(afs)

hh$y = (hh$y/mx) * diff(range(afs))+min(afs)
 
 
plot(range(xh), range(afs), axes = FALSE, type = 'n')
 
polygon(c(min(fh$x), fh$x, max(fh$x)), c(0, fh$y, 0), 

		col = '#FF000044', border = NA)

polygon(c(min(hh$x), fh$x, max(hh$x)), c(0, hh$y, 0), 

		col = '#0000FF44', border = NA)

polygon(c(min(hh$x), hh$x, max(hh$x)), c(0, hh$y, 0), 

		col = '#0000FF44', border = NA)		

polygon(c(min(fh$x), fh$x, max(fh$x)), c(0, hh$y, 0), 

		col = '#FF000044', border = NA)
 
rrs[is.na(rrs)] = 1
 
lines(x, afs)

lines(x, smooth.spline(rrs, df = 10)$y, lty = 4)

axis(2)

axis(1, labels = labels, at = at)
 
lines(trans(c(BA, BA)), c(-9E9, 9E9), lty = 2)
 
plot(range(hh$x), range(hh$y),ylim = rev(range(hh$y)), type = 'n', axes = FALSE, xlab = '', ylab = '')

polygon(c(min(fh$x), hh$x, max(fh$x)), c(0, hh$y, 0), 

		col = '#0000FF99', border = NA)
 
lines(rep(trans(BA), 2), c(-9E9, 9E9), lty = 2)
 
 
 
