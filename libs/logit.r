
logit <- function(r) {
    r[r < 0.0000001] = 0.0000001
    log(r/(1-r))
}

