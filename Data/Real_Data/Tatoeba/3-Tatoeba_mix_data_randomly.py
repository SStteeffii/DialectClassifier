import random
import csv

with open('../Delimiters/Delimiter_NewLine.txt', 'r', encoding='utf-8') as file:
    delimiter_newline = file.read()

with open('../Delimiters/Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
    delimiter_tab = file.read()

# Namen der Quelldateien
files = ['bar_Tatoeba_labeled_resized.tsv', 'nds_Tatoeba_labeled_resized.tsv', 'de_Tatoeba_labeled_resized.tsv']


def mix(files, filen):

    all_rows = []

    # read all lines of all files
    for filename in files:
        with open(filename, 'r', encoding='utf-16', newline='') as infile:
            reader = csv.reader(infile, delimiter=delimiter_newline)
            all_rows.extend(list(reader))

    # mix lines randomly
    random.shuffle(all_rows)

    with open(filen, 'w', encoding='utf-16') as outfile:
        writer = csv.writer(outfile, delimiter=delimiter_newline)
        writer.writerows(all_rows)

# mix(files, Tatoeba_de+nds+bar_mixed.tsv)
