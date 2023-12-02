import csv
import sys
import time

with open('Delimiter_NewLine.txt', 'r', encoding='utf-8') as file:
    delimiter_newline = file.read()

with open('Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
    delimiter_tab = file.read()

files = [['bar_MrDialekt.tsv'], ['nds_MrDialekt.tsv'], ['de_Wiki_cleaned_resized_NewLine.tsv']]

for file in files:
    filename = file[0]
    with open(filename, 'r', encoding='utf-16') as infile:
        reader = csv.reader(infile, delimiter=delimiter_newline)
        for row in reader:
            file.extend(row)

# Determine the target size as the size of the smallest file
target_size = min(len(files[0]), len(files[1]), len(files[2]))  # = 36256

for file in files:
    filename = file[0]
    data = file[1:]
    current_size = len(data)

    if current_size <= target_size:  # smallest file == target_size
        resized_data = data
    else:
        resized_data = data[:target_size]

    with open(filename[:-4] + '_resized.tsv', 'w', encoding='utf-16') as result_file:
        csv.writer(result_file, delimiter=delimiter_newline).writerow(resized_data)
