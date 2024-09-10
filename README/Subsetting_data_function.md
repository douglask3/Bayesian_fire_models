# subset_function - functions for constraining data
There are several functions for constraining data at runtime, both spacially and temporally.

## *AR6 regions* 
Function: `ar6_region`
Inputs: `region_code`
When opening data, setting ''Country'' or ''Continent'' will constrain the extent to that country or continent, and mask areas outside of it. Uses Natural Earth. If you define a country, it one look at the continent. Use None is you don't want any. Continent options are:
* 'South America'
* 'Oceania'
* 'Europe'
* 'Afria'
* 'North America'
* 'Asia'

## *Ecoregions*
Function: `constrain_olson`
Inputs: `ecoregions`
''ecoregions'' is a numeric list (i.e [3, 7, 8]) where numbers pick Olson bomes and mask out everywhere else. If you  pick more than one, it returns a map of all of them.
* **None** Return all areas.
* **1** Tropical and subtropical moist broadleaf forests
* **2** Tropical and subtropical dry broadleaf forests
* **3** Tropical and suptropical coniferous forests
* **4** Temperate broadleaf and mixed forests
* **5** Temperate Coniferous Forest
* **6** Boreal forests / Taiga
* **7** Tropical and subtropical grasslands, savannas and shrublands
* **8** Temperate grasslands, savannas and shrublands
* **9** Flooded grasslands and savannas
* **10** Montane grasslands and shrublands
* **11** Tundra
* **12** Mediterranean Forests, woodlands and scrubs
* **13** Deserts and xeric shrublands
* **14** Mangroves

## *Brazillian legal Biomes*
Function: `constrain_BR_biomes`
Inputs: `biome_ID`

''biome_ID'' is a numeric list where numbers pick Brazilian biomes and mask out everywhere else. If you pick more than one, it returns a map of all of them.

* **1** Amazonia
* **2** Caatinga
* **3** Cerrado
* **4** Atlantic Forest
* **5** Pampa
* **6** Pantanal

## *GFED Regions*
Function: `constrain_GFED`
Inputs: `region`

Constrains data to GFED region(s):
region -- numeric list (i.e [3, 7, 8]) where numbers pick GFED region.
            You can pick more than one:
            1 BONA
            2 TENA
            3 CEAM
            4 NHSA
            5 SHSA
            6 EURO
            7 MIDE
            8 NHAF
            9 SHAF
            10 BOAS
            11 CEAS
            12 SEAS
            13 EQAS
            14 AUST

## *Contrain to political regions*
Function: `constrain_natural_earth`
Inputs: `Country` or `Continent`
Uses natural Earth to continent or country. See [www.naturalearthdata.com](www.naturalearthdata.com) for info

## *To  range of years*
Function: `sub_year_range`
Inputs: `year_range`

## *To months of year*
Function: `sub_year_months`
Inputs: `months_of_year`
