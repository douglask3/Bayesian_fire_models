# subset_function - functions for constraining data
There are several functions for constraining data at runtime, both spacially and temporally.

## *AR6 regions* 
Function: `ar6_region`
Inputs: `region_code`


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
When opening data, setting ''Country'' or ''Continent'' will constrain the extent to that country or continent, and mask areas outside of it. Uses Natural Earth. If you define a country, it wont look at the continent. Use None is you don't want any. Continent options are:
* 'South America'
* 'Oceania'
* 'Europe'
* 'Afria'
* 'North America'
* 'Asia'

Country options are:

'Afghanistan', 'Albania', 'Algeria', 'Angola', 'Antarctica', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bangladesh', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herz.', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Central African Rep.', 'Chad', 'Chile', 'China', 'Colombia', 'Congo', 'Costa Rica', 'Croatia', 'Cuba', 'Cyprus', 'Czechia', "Côte d'Ivoire", 'Dem. Rep. Congo', 'Denmark', 'Djibouti', 'Dominican Rep.', 'Ecuador', 'Egypt', 'El Salvador', 'Eq. Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Falkland Is.', 'Fiji', 'Finland', 'Fr. S. Antarctic Lands', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Greenland', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Mali', 'Mauritania', 'Mexico', 'Moldova', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'N. Cyprus', 'Namibia', 'Nepal', 'Netherlands', 'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'North Macedonia', 'Norway', 'Oman', 'Pakistan', 'Palestine', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'S. Sudan', 'Saudi Arabia', 'Senegal', 'Serbia', 'Sierra Leone', 'Slovakia', 'Slovenia', 'Solomon Is.', 'Somalia', 'Somaliland', 'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States of America', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam', 'W. Sahara', 'Yemen', 'Zambia', 'Zimbabwe', 'eSwatini'

## *Contrain to states*
Comeing soon! We'll have an option to constrain ConFire to states. Here are the options we'll have:

*admin_1_states_provinces*:

'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'

*admin_0_map_units*:

'Afghanistan', 'Albania', 'Algeria', 'Angola', 'Antarctica', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bangladesh', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herz.', 'Botswana', 'Bougainville', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Central African Rep.', 'Chad', 'Chile', 'China', 'Colombia', 'Congo', 'Costa Rica', 'Croatia', 'Cuba', 'Cyprus', 'Czechia', "Côte d'Ivoire", 'Dem. Rep. Congo', 'Denmark', 'Djibouti', 'Dominican Rep.', 'Ecuador', 'Egypt', 'El Salvador', 'England', 'Eq. Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Falkland Is.', 'Fiji', 'Finland', 'Fr. S. Antarctic Lands', 'France', 'French Guiana', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Greenland', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Mali', 'Mauritania', 'Mexico', 'Moldova', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'N. Cyprus', 'N. Ireland', 'Namibia', 'Nepal', 'Netherlands', 'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'North Macedonia', 'Norway', 'Oman', 'Pakistan', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'S. Sudan', 'Saudi Arabia', 'Scotland', 'Senegal', 'Serbia', 'Sierra Leone', 'Slovakia', 'Slovenia', 'Solomon Is.', 'Somalia', 'Somaliland', 'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Svalbard Is.', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United States of America', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam', 'W. Sahara', 'Wales', 'West Bank', 'Yemen', 'Zambia', 'Zimbabwe', 'eSwatini'

## *To  range of years*
Function: `sub_year_range`
Inputs: `year_range`

## *To months of year*
Function: `sub_year_months`
Inputs: `months_of_year`
