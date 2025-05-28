#!/bin/bash

BASE_URL="https://opendap.uea.ac.uk/opendap/hyrax/firecrew/VCF_MOD44B_250m/"
LIST_URL="${BASE_URL}"

# Step 1: Get the list of date-based directories
dates=$(wget -qO- "$LIST_URL" | grep -oE '[0-9]{4}\.[0-9]{2}\.[0-9]{2}/' | sed 's|/$||' | sort -u)

# Step 2: Loop through each date directory and download files
for date in $dates; do
    echo "Downloading files from: $BASE_URL$date/"
    wget -r -np -nH --cut-dirs=3 -A ".hdf" "${BASE_URL}${date}/"
done

