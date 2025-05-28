sourceAllLibs <- function(path ='libs/', trace = TRUE, recursive = TRUE,
                          stopIfError = TRUE, ...) {
                              
    files = list.files(path, pattern = "\\.[RrSsQq]$", recursive = recursive)
	for (nm in files) {
		if(trace) cat(nm, ":")

		if (stopIfError) source(file.path(path, nm), ...)
			else try(source(file.path(path, nm), ...))

		if(trace) cat("\n")
	}
}
