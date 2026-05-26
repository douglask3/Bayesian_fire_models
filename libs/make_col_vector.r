make_col_vector <- function(r,g,b,ncols,
							limits = NULL, whiteAt0 = TRUE) {
    
	if (!is.null(limits)) {
		ncols = length(limits+1)
		if (whiteAt0 && class(r)=="character") {
			whiteIndex = which(r == "white" | r == "#FFFFFF")
			if (length(whiteIndex) == 0)  whiteIndex = which(r == "black" | r == "#000000")
			if (length(whiteIndex) != 0) {
				zeroIndex  = which(limits[-1]>0 & head(limits,-1)<0)

				if ( length(zeroIndex)==0) {
					if (limits[1] > 0)r = r[whiteIndex:length(r)]
						else r = r[1: whiteIndex]

					return(make_col_vector_gubbins(r,ncols = length(limits)))
				}

				negCols = make_col_vector_gubbins(r[1:whiteIndex],
												  ncols = zeroIndex+1)
				posCols = make_col_vector_gubbins(r[whiteIndex:length(r)],
												  ncols = length(limits) - zeroIndex + 1)
				return(c(negCols,posCols[-1]))
				}
			}
	}
	return(make_col_vector_gubbins(r, g, b, ncols) )
}

make_col_vector_gubbins <- function(r, g, b, ncols) {
    library(colorspace)
	if (class(r)=="character") {
		col=col2rgb(r)
                
		r=col[1,]/255
		g=col[2,]/255
		b=col[3,]/255
	}

	col_vec <- function(a) approx(seq(1,ncols,length.out=length(a)),a,1:ncols)$y

	cols=colorspace::RGB(col_vec(r),col_vec(g),col_vec(b))

	return(hex(cols))
}
