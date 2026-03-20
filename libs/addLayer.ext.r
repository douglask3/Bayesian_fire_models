addLayer.ext <- function(a,b) {
	
	if(!extent(a)==extent(b)) {
		a=crop(a,b)
		b=crop(b,a)
	}
	return(addLayer(a,b))
}