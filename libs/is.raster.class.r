is.raster <- function(...) is.raster.class(...)

is.raster.class <- function(r) {
	classi=class(r)
	if (classi=="Raster" || classi=="RasterStack" || classi=="RasterBrick" ||
	    classi=="RasterLayer") return(TRUE)
	return(FALSE)
}