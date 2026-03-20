layer.apply <- function(indexs,FUN,...) {
 	if (class(FUN)!="function") FUN <- match.fun(FUN)

 	if (is.raster.class(indexs)) {
		x=FUN(indexs[[1]],...)
		if (!is.raster.class(x))  x=list(x)
		if (nlayers(indexs)>1) {
			if (is.raster.class(x)) {
				for (i in 2:nlayers(indexs)) x=addLayer.ext(x,FUN(indexs[[i]],...))
			} else {
				for (i in 2:nlayers(indexs)) x[[i]]=FUN(indexs[[i]],...)
			}
		}
	} else {
		if (class(indexs) == "list") index1 = indexs[[1]]
            else index1 = indexs[1]
		x=FUN(index1,...)
		for (i in indexs[-1]) {
			y=FUN(i,...)
			if (!is.null(y)) {
				if (is.null(x)) x=y else x=addLayer.ext(x,y)
			}
		}
 	}
 	return(x)
}
