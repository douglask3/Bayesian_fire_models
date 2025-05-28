multiNoFileWarnings <- function(...) {
    oneFileWarning(...)
    noFileWarning(...)
}

oneFileWarning <- function(x, varName)
    if (length(x) > 1) cat(paste('\nWARNING: None-unique file for ', 'CLM',
                               ' - variable:', varName ,
                               '. First file selected\n', sep = ''))


noFileWarning <- function(x, varName) {
    if (length(x) == 0) {
        cat(paste('\nWARNING: No file found for - variable:', varName,
                       '. Model ignored for this variable\n'))
        return(TRUE)
    }
    return(FALSE)
}
