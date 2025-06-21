library(poibin)
dir = 'Amazon/'
event = 0.526584627
ssps = c('historical', 'ssp126', 'ssp370', 'ssp585')

models = c("GFDL-ESM4-", "IPSL-CM6A-LR-", 
		   "MPI-ESM1-2-HR-", "MRI-ESM2-0-", "UKESM1-0-LL-"	)
		   
atleast_n_events <- function(pevent, n) {
	1 - ppoibin(n - 1, pevent)
#	combs <- combn(length(pevent), n, simplify = FALSE)
#	sum(sapply(combs, function(idx) {
#		prod(pevent[idx]) * prod(1 - pevent[-idx])
#	}))
}

cal_occurance_prob_for_ssp <- function(ssp, model, dir,
								       event = NaN, nyrs = 78, whichend = 'end', nevents = 1) {
	dat = read.csv(paste(dir, ssp, model, 
				   'mean/points-Evaluate.csv', sep = '/')	, stringsAsFactors = FALSE)[,-1]
	print(ssp)
	print(model)
	print(nevents)
	
	before25 = substr(colnames(dat), 2, 5) < 2025
	if (is.na(event))
		event =  quantile(as.matrix(dat[,before25]), 1-1/12000)
	
	if (whichend == 'end') dat = dat[,!before25]
	pevent =  apply(dat, 2, function(x) mean(x>event))
	
	if (length(pevent) < (nyrs * 12)) {
		if (whichend == 'end') {
			pevent = c(pevent, 
					   rep(tail(pevent, 13), 
					       length.out = (nyrs * 12) - length(pevent)))
		} else {
			pevent = rep(pevent, length.out = (nyrs * 12))
		}
	}	
	if (nevents == 1) {
		out = 1-prod(1-pevent)
	} else {
		out = atleast_n_events(pevent, nevents)
	}
	return(c(event, out))
}

cal_occurance_prob_for_model <- function(event = NaN, ...) {
	
	houts = cal_occurance_prob_for_ssp(ssps[1], whichend = 'start', 
								       event = event, ...)
	event = houts[1]
	
	fouts = sapply(ssps[-1], cal_occurance_prob_for_ssp, event = event, ...)
	
	return(c(houts, fouts[2,]))
}

outs1 = sapply(models, cal_occurance_prob_for_model, dir, event = event)
outs2 = sapply(models, cal_occurance_prob_for_model, dir, 
			event = event, nevents = 2)
outs3 = sapply(models, cal_occurance_prob_for_model, dir, 
			   event = event, nevents = 3)
			   
		   
