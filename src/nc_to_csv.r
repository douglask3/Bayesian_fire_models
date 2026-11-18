
files = c("outputs/Chilean_Temperate_Forests_and_Matorral-AF.nc", 
          "outputs/Chilean_Temperate_Forests_and_Matorral-fire_mask.nc", 
          "outputs/Chilean_Temperate_Forests_and_Matorral-Likelihood.nc", 
          "outputs/Midwestern_Canadian_Shield_forests-AF.nc", 
          "outputs/Midwestern_Canadian_Shield_forests-fire_mask.nc", 
          "outputs/Midwestern_Canadian_Shield_forests-Likelihood.nc", 
          "outputs/Northwest_Iberia-AF.nc", 
          "outputs/Northwest_Iberia-fire_mask.nc", 
          "outputs/Northwest_Iberia-Likelihood.nc", 
          "outputs/Northwest_Iberia-liklihood.nc")

open2csv <- function(file) {
    r = raster(file)
    out = cbind(xyFromCell(r, 1:length(r[])), r[])
    out = out[out[,3]<9E9,]
    out_file = gsub('.nc','.csv', file)
    colnames(out) = c('longitude', 'latitude', 'value')
    write.csv(out, out_file, row.names = FALSE)
}

sapply(files, open2csv)

