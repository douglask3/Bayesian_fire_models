graphics.off()
library(poibin)
dirs = c('Amazon', 'Pantanal', 'LA', 'Congo')
region_names = c('Northeast Amazon', 'Pantanal and Chiquitano', 'Southern California', 'Congo Basin')

events = c(0.526584627/100, 9.289187576/100, 0.634514248/100, 0.471739/100)
events = c(43.2328163/100, 75.59482066/100, 3.554378271/10, 36.43572165/100)
mnths = c(1, 8, 1, 7)
life_expectancys = c(76, 76, 80, 62)
ssps = c('historical', 'ssp126', 'ssp370', 'ssp585')

models = c("GFDL-ESM4-", "IPSL-CM6A-LR-", 
		   "MPI-ESM1-2-HR-", "MRI-ESM2-0-", "UKESM1-0-LL-")
		   
atleast_n_events <- function(n, pevent)  {
	if (n == 1) 
		out = 1-prod(1-pevent)
	else
		out = 1 - ppoibin(n - 1, pevent)
	return(out)
}


cal_occurance_prob_for_ssp <- function(ssp, model, dir,
								       event = 0, life_expectancy = 78, whichend = 'end', nevents = 1:3, mnth = NaN) {
	print(ssp)
	print(model)
	print(nevents)
							
	tfile = paste0('temp/occurance_prob', 
				   tail(strsplit(dir, '/')[[1]], 1), ssp, model, round(event, 4), mnth, '3.Rd')
	
	if (file.exists(tfile)){#	&& dir != 'Congo') {
		load(tfile)
		return(c(event, outs))
	}
	dat = read.csv(paste(dir, ssp, model, 
				   'pc-95.0/points-Evaluate.csv', sep = '/')	, stringsAsFactors = FALSE)[,-1]
	
	if (!is.na(mnth))
		dat = dat[, seq(mnth, ncol(dat), by = 12)]
	before25 = substr(colnames(dat), 2, 5) < 2025
	if (event == 0)
		event =  quantile(as.matrix(dat[,before25]), 1-1/12000)
	#if (dir == 'Congo')
	#		browser()
	if (whichend == 'end') dat = dat[,!before25]
	
	#browser()
	#for_decade <- function(yr) {
#		ddat = dat[,yr:(yr+9)]
#		apply(ddat, 1, function(x) any(x>event))
#	}
	#dat = sapply(seq(1, ncol(dat)-10), for_decade)
	pevent =  apply(dat, 2,  function(x) mean(x>event))
	
	if (is.na(mnth)) {
		scale = 12
		pevent = pevent / 12
	} else { 
		scale = 1
	}
	
	if (length(pevent) < (life_expectancy * scale)) {
		if (whichend == 'end') {
			pevent = c(pevent, 
					   rep(tail(pevent, scale), 
					       length.out = (life_expectancy * scale) - length(pevent)))
		} else {
			pevent = rep(pevent, length.out = (life_expectancy * scale))
		}
	}	
	
	outs = sapply(nevents, atleast_n_events,  pevent)
	
	save(event, outs, file = tfile)
	return(c(event, outs))
}

cal_occurance_prob_for_model <- function(event = NaN, ...) {
	
	houts = cal_occurance_prob_for_ssp(ssps[1], whichend = 'start', 
								       event = event, ...)
	event = houts[1]
	
	fouts = sapply(ssps[-1], cal_occurance_prob_for_ssp, event = event, ...)
	
	return(100*cbind(houts, fouts)[-1,])
}

for_region <- function(name, life_expectancy, xaxist =  T, yaxist = T,...) {
	outs = lapply(models, cal_occurance_prob_for_model, 
				  life_expectancy = life_expectancy, ...)
	outs = lapply(1:length(ssps), function(i) 
				  sapply(outs, function(out) out[,i]))
	xrange = range(unlist(outs))
	plot(xrange, c(0.1, length(ssps)), type ='n', xlab = '',
		 axes = FALSE, ylab = '')
	polygon(c(-9E9, 9E9, 9E9, -9E9), c(0, 0, 1, 1)+0.1, border = "grey", lty = 2, col = '#dddddd')
	polygon(c(-9E9, 9E9, 9E9, -9E9), c(0, 0, 1, 1)+2.1, border = "grey", lty = 2, col = '#dddddd')
	grid(nx = NULL,
     ny = NA)
	box()
	plot_ssp <- function(out, y1) {
		plot_return <- function(y2) {
			col = c("#f68373", "#c7384e", "#862976")[y2]
			xs = out[y2,]
			#xs = c(0, 100)
			l1 = y1 +0.8*(((y2-1) /nrow(out))-1)
			l2 = y1 +0.9*((y2 /nrow(out))-1)
			polygon(range(xs)[c(1, 2, 2, 1)], c(l1, l1, l2, l2), 
					col = paste0(col, 'BB'), border = col)
			lapply(xs, function(x) lines(c(x, x), c(l1, l2), col = 'black', lwd = 2))
			tx = range(xs)
			ty = 0.5*l1 + 0.5*l2
			txt = unique(round(tx))
			text(x = tx[1], y = ty, txt[1], adj = 1.4, cex = 0.67)
			text(x = tx[2], y = ty, txt[2], adj = -0.4, cex = 0.67)
		}
		lapply(1:nrow(out), plot_return)
	}
	mapply(plot_ssp, outs, 1:length(outs))
	
	
	if (xaxist) axs =2 else axs = 4
	axis(axs, at = 0.5, labels = paste0('Born in ', 2025-life_expectancy ), las = 2)
	axis(axs, at = 1.5, labels = '\nstrong\nmitigation-\nssp0126\n', las = 2)
	axis(axs, at = 2.5, labels = paste0('middle of\nthe road-\nssp270'), las = 2) # \n2025 to ', 2025+life_expectancy 
	axis(axs, at = 3.5, labels = 'no\nmitigation\nssp585\n', las = 2)
	
	if (yaxist) axis(1)
	#title(paste0(letters[which(region_names == name)], ') ', name), line = 0.25, xpd = NA)
	title( name, line = 0.25, xpd = NA)
}
	
#for_region <-function(...) {
#	outs = lapply(1:3, for_returns, ...)
#	xrange = range(unlist(outs))
#	plot(xrange, c(0, length(ssps)))
#	browser()
#}
png("life_return.png", height = 5.1, width = 7.4, units = 'in', res = 300) 
layout(rbind(matrix(1:4, ncol = 2), 5), heights = c(1, 1, 0.28))
par(mar = c(1.5, 0.5, 0.5, 0.5), oma = c(2.5, 6, 1, 6))
outs1 = mapply(for_region, region_names, life_expectancys, dirs, event = events,  mnth = mnths, 
				xaxist= c(T, T, F, F), yaxist = c(F,T, F,T))

mtext(side = 1, outer = TRUE, 'Likelihood (%)', line = -3.2)
par(mar = c(00, 0, 2.2, 0))
plot.new()
legend('bottom', horiz = TRUE, pch= 15, pt.cex = 3, col = paste0(c("#f68373", "#c7384e", "#862976"), 'bb'), c('Once', 'Twice', 'Three times'), bty = 'n', xpd = NA)
			   
dev.off()

#events = c(0, 0, 0, 0)
#png("life_return_1_in_1000.png", height = 5.5*1.3, width = 4.5*1.3, units = 'in', res = 300) 
#par(mfrow = c(length(dirs),1), mar = c(1.5, 3.2, 2, 0.5), oma = c(2.5, 0, 0, 0))
#outs2 = mapply(for_region, region_names, life_expectancys, dirs, event = events,  mnth = mnths)
#mnths = c(NaN, NaN, NaN, NaN)
#mtext(side = 1, outer = TRUE, 'Likelihood (%)', line = 1)
#			   
#dev.off()
