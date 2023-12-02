import random
import csv

with open('Delimiter_NewLine.txt', 'r', encoding='utf-8') as file:
    delimiter_newline = file.read()

with open('Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
    delimiter_tab = file.read()

files = ['data_bar_cleaned_resized_preprocessed.tsv', 'data_nds_cleaned_resized_preprocessed.tsv', 'data_de_cleaned_resized_preprocessed.tsv']

all_rows = []

# read all lines of all files
for filename in files:
    with open(filename, 'r', encoding='utf-16') as infile:
        reader = csv.reader(infile, delimiter=delimiter_newline)
        all_rows.extend(list(reader))

# mix lines randomly
random.shuffle(all_rows)

with open('Wikipedia_de+nds+bar_mixed.tsv', 'w', encoding='utf-16') as outfile:
    writer = csv.writer(outfile, delimiter=delimiter_newline)
    writer.writerows(all_rows)
