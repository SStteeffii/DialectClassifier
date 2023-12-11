import csv

with open('../Delimiters/Delimiter_NewLine.txt', 'r', encoding='utf-8') as file:
    delimiter_newline = file.read()

with open('../Delimiters/Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
    delimiter_tab = file.read()

files = ['../Data_Training/Dialektversum_de+nds+bar_mixed_preprocessed.tsv',
         '../Data_Training/MrDialect_de+nds+bar_mixed_preprocessed.tsv',
         '../Data_Training/Wikipedia_de+nds+bar_mixed.tsv']
# files = ['../Data_FinalTest/Tatoeba_de+nds+bar_mixed_preprocessed.tsv'] #  with 100 % "test"
# files = ['../Data_FinalTest/Wikipedia_de+nds+bar_mixed_forWordbooks.tsv'] #  with 100 % "test"


def splitlabel(filenames):
    for file in filenames:
        count_rows = 0
        row_count = 0

        with open(file, 'r', encoding='utf-16') as infile:
            tsv_reader = csv.reader(infile, delimiter=delimiter_newline)
            for row in tsv_reader:
                count_rows += 1

        with open(file, 'r', encoding='utf-16') as infile:
            tsv_reader = csv.reader(infile, delimiter=delimiter_newline)
            for row in tsv_reader:
                for r in row:
                    row_count += 1
                    percentage = (row_count*100)/count_rows
                    if percentage <= 80:
                        splitlabel = "train"
                    elif percentage <= 90:
                        splitlabel = "validate"
                    elif percentage <= 100:
                        splitlabel = "test"

                    with open(file[:-4] + '_splitlabel.tsv', 'a', encoding='utf-16') as tsv_file:
                        tsv_file.write(splitlabel + delimiter_tab + r + delimiter_newline)

# splitlabel(files)