import random
import csv

with open('Delimiter_NewLine.txt', 'r', encoding='utf-8') as file:
    delimiter_newline = file.read()

with open('Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
    delimiter_tab = file.read()

files = ['bar_Dialektversum-RespektEmpire_resized.tsv', 'de_Wiki_cleaned_resized_NewLine_resized.tsv', 'nds_Dialektversum-Oeverstetter_resized.tsv']

all_rows = []

for filename in files:
    with open(filename, 'r', encoding='utf-16') as infile:
        reader = csv.reader(infile, delimiter=delimiter_newline)
        for row in reader:
            if row:
                labeled_row = str(filename[:filename.find("_")] + delimiter_tab + str(row[0]))
                all_rows.append([labeled_row])

# mix lines randomly
random.shuffle(all_rows)

with open('Dialektversum_de+nds+bar_mixed.tsv', 'w', encoding='utf-16') as outfile:
    writer = csv.writer(outfile, delimiter=delimiter_newline)
    writer.writerows(all_rows)
