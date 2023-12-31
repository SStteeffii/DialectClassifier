import random
import csv

with open('../Delimiters/Delimiter_NewLine.txt', 'r', encoding='utf-8') as file:
    delimiter_newline = file.read()

with open('../Delimiters/Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
    delimiter_tab = file.read()

data = ['bar_Dialektversum-RespektEmpire_resized.tsv', 'de_Wiki_cleaned_resized_NewLine_resized.tsv', 'nds_Dialektversum-Oeverstetter_resized.tsv']


def mix(files, filen):

    all_rows = []

    for filename in files:
        dialect = str(filename[filename.rfind("/")+1:filename.find("_")])
        with open(filename, 'r', encoding='utf-16') as infile:
            reader = csv.reader(infile, delimiter=delimiter_newline)
            for row in reader:
                if row:
                    labeled_row = str(dialect + delimiter_tab + str(row[0]))
                    all_rows.append([labeled_row])

    # mix lines randomly
    random.shuffle(all_rows)

    with open(filen, 'w', encoding='utf-16') as outfile:
        writer = csv.writer(outfile, delimiter=delimiter_newline)
        writer.writerows(all_rows)

# mix(data, Dialektversum_de+nds+bar_mixed.tsv)
