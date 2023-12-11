import csv

with open('Delimiter_NewLine.txt', 'r', encoding='utf-8') as file:
    delimiter_newline = file.read()

with open('Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
    delimiter_tab = file.read()

files = ['bar_Tatoeba_sentences.tsv', 'de_Tatoeba_sentences.tsv', 'nds_Tatoeba_sentences.tsv']

for file in files:

    rows = []

    with open(file, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter=delimiter_tab)
        rows = [row[2] for row in reader]

    with open(str(file[:-14] + '_labeled.tsv'), 'w', encoding='utf-16') as outfile:
        for row in rows:
            outfile.write(str(file[:file.find("_")]) + delimiter_tab + row + delimiter_newline)
