# Function to convert a number to hexadecimal
number_to_hex <- function(i, j = 1) {
    
    i = (i-1)/(nlevs-1)
    j = (j-1)/(nlevs-1)
    number =  (i+j)/2#sqrt(i^2 + j^2)
    #number = min(i, j)
    if (number > 1) number = 1
    number = round(number*255)

  if (number < 0 || number > 255) {
    stop("Number must be between 0 and 255")
  }
  hex <- sprintf("%02X", number)
  return(hex)
}
