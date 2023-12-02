import random
import csv

with open('Delimiter_NewLine.txt', 'r', encoding='utf-8') as file:
    delimiter_newline = file.read()

with open('Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
    delimiter_tab = file.read()

file = 'de_Wiki_cleaned_resized_Tab.tsv'

all_rows = []

with open(file, 'r', encoding='utf-16') as infile:
    reader = csv.reader(infile, delimiter=delimiter_tab)
    all_rows.extend(reader)


with open('de_Wiki_cleaned_resized_NewLine.tsv', 'w', encoding='utf-16') as outfile:
    writer = csv.writer(outfile, delimiter=delimiter_newline)
    writer.writerows(all_rows)
