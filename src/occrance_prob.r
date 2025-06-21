graphics.off()
library(poibin)
dirs = region_name = c('Amazon', 'Pantanal', 'LA')
events = c(0.526584627, 9.289187576/10, 0.634514248)
mnths = c(NaN, 8, 1)
life_expectancys = c(76, 76, 80)
ssps = c('historical', 'ssp126', 'ssp370', 'ssp585')

models = c("GFDL-ESM4-", "IPSL-CM6A-LR-", 
		   "MPI-ESM1-2-HR-", "MRI-ESM2-0-", "UKESM1-0-LL-"	)
		   
atleast_n_events <- function(n, pevent)  {
	if (n == 1) 
		out = 1-prod(1-pevent)
	else
		out = 1 - ppoibin(n - 1, pevent)
	return(out)
}


cal_occurance_prob_for_ssp <- function(ssp, model, dir,
								       event = NaN, life_expectancy = 78, whichend = 'end', nevents = 1:3, mnth = NaN) {
	print(ssp)
	print(model)
	print(nevents)
							
	tfile = paste0('temp/occurance_prob', 
				   tail(strsplit(dir, '/')[[1]], 1), ssp, model, '3.Rd')
	if (file.exists(tfile) && FALSE) {
		load(tfile)
		return(c(event, outs))
	}
	dat = read.csv(paste(dir, ssp, model, 
				   'mean/points-Evaluate.csv', sep = '/')	, stringsAsFactors = FALSE)[,-1]

	if (!is.na(mnth))
		dat = dat[, seq(mnth, ncol(dat), by = 12)]
	before25 = substr(colnames(dat), 2, 5) < 2025
	if (is.na(event))
		event =  quantile(as.matrix(dat[,before25]), 1-1/12000)
	
	if (whichend == 'end') dat = dat[,!before25]
	
	pevent =  apply(dat, 2, function(x) mean(x>event))
	
	if (length(pevent) < (life_expectancy * 12)) {
		if (whichend == 'end') {
			pevent = c(pevent, 
					   rep(tail(pevent, 13), 
					       length.out = (life_expectancy * 12) - length(pevent)))
		} else {
			pevent = rep(pevent, length.out = (life_expectancy * 12))
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

for_region <- function(name, life_expectancy, ...) {
	outs = lapply(models, cal_occurance_prob_for_model, 
				  life_expectancy = life_expectancy, ...)
	outs = lapply(1:length(ssps), function(i) 
				  sapply(outs, function(out) out[,i]))
	xrange = range(unlist(outs))
	plot(xrange, c(0, length(ssps)), type ='n', xlab = '',
		 axes = FALSE, ylab = '')
	plot_ssp <- function(out, y1) {
		plot_return <- function(y2) {
			col = c("#f68373", "#c7384e", "#862976")[y2]
			xs = out[y2,]	
			l1 = y1 -1.0 + (y2-1) /nrow(out)
			l2 = y1 -1.0 + (y2) /nrow(out)
			polygon(range(xs)[c(1, 2, 2, 1)], c(l1, l1, l2, l2), 
					col = paste0(col, '44'), border = col)
			lapply(xs, function(x) lines(c(x, x), c(l1, l2), col = col))
		}
		lapply(1:nrow(out), plot_return)
	}
	mapply(plot_ssp, outs, 1:length(outs))
	axis(2, at = 0.5, labels = paste0('Born in\n', 2025-life_expectancy ))
	axis(2, at = 1.5, labels = 'ssp126\n')
	axis(2, at = 2.5, labels = paste0('ssp270\nto', 2025+life_expectancy ))
	axis(2, at = 3.5, labels = 'ssp585\n')
	axis(1)
	title(name)
}
	
#for_region <-function(...) {
#	outs = lapply(1:3, for_returns, ...)
#	xrange = range(unlist(outs))
#	plot(xrange, c(0, length(ssps)))
#	browser()
#}

par(mfrow = c(length(dirs),1))
outs1 = mapply(for_region, region_name, life_expectancys, dirs, event = events,  mnth = mnths)

mtext(side = 1, outer = TRUE, 'Likelihood (%)')
			   
		   
