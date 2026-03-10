## Datasets

* For "State of Wildfires 2023/24": https://doi.org/10.5281/zenodo.11420743


### Making State of Wildfire datasets

#### ISIMIP

Do this:
1. Download isimip data, either via isimip portal or by running the script ```make_inputs/isimip/get_files_from_isimip_repo.sh```.
* If from portal, the files you need are listed in ``make_inputs/isimip/filelist.txt```
* If running  the script:
a. open get_files_from_isimip_repo.sh and change the directory listed to the directory you wish to store the data.
b. Run:

```
cd make_inputs/isimip/
sh get_files_from_isimip_repo.sh
```

