logistic <- function(r) {
    xs = 1/(1+exp(r*(-1)))
    xs = (1.000002) *xs - 0.000001
    return(xs)
}

