library(raster)

example_dir = 'data/data/driving_data2425/Amazon/nrt/era5_monthly-old/CF/pr/'
output_dir = 'data/data/driving_data2425/Amazon/nrt/era5_monthly/CF/pr/'
input_dir = 'data/data/driving_data2425/Amazon/nrt/pr_old/'
varnames = c('precip', 'dry_days', 'max_consec_dry')

for_variable <- function(varname) {
    eg_file = brick(paste0(example_dir, varname, '1.nc'))
    in_files = list.files(input_dir)
    in_files = in_files[grepl(varname, in_files)]
    
    for_file <- function(file) {
        input = brick(paste0(input_dir, file))
        eg_file_i = eg_file
        
        eg_file_i[] = input[]
        writeRaster(eg_file_i, paste0(output_dir, file), overwrite=TRUE)
        
    }
    sapply(in_files, for_file)
}
sapply(varnames, for_variable)
