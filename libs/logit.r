
logit <- function(r) {
    r = (r+0.000001)/1.000002
    log(r/(1-r))
}



