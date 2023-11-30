import csv

with open('../Delimiters/Delimiter_NewLine.txt', 'r', encoding='utf-8') as file:
    delimiter_newline = file.read()

with open('../Delimiters/Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
    delimiter_tab = file.read()


def read_data(filename):
    data = []

    with open(filename, 'r', encoding='utf-16') as infile:
        tsv_reader = csv.reader(infile, delimiter=delimiter_newline)
        for row in tsv_reader:
            for r in row:
                dialect, datastring = r.split(delimiter_tab)
                data.append(datastring)
    return data


def read_data_with_labels(filename):
    data = []
    labels = []
    splitlabels = []
    with open(filename, 'r', encoding='utf-16') as infile:
        tsv_reader = csv.reader(infile, delimiter=delimiter_newline)
        for row in tsv_reader:
            for r in row:
                tab_count = r.count(delimiter_tab)
                if tab_count >= 2:
                    splitlabel, dialect, datastring = r.split(delimiter_tab, 2)
                    splitlabels.append(splitlabel)
                    labels.append(dialect)
                    data.append(datastring)
    return splitlabels, data, labels

# print(read_data_with_labels('data_mixed_labeled_preprocessed.tsv'))
