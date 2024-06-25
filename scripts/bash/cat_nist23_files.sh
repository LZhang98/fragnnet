#!/bin/bash

cd ~/Downloads/NIST23/NIST23
nist_dset=$1

# Specify the file extension you're looking for
file_extension=$2

# Create an array to store the file names
file_list=()

# Populate the array with file names of the specified extension
for file in ${nist_dset}/*"$file_extension"; do
    if [ -f "$file" ]; then
        file_list+=("$file")
    fi
done

# Concatenate the files using cat command
if [ ${#file_list[@]} -gt 0 ]; then
    cat "${file_list[@]}" > ${nist_dset}${file_extension}
    echo "Concatenation complete."
else
    echo "No files with the specified extension found."
fi