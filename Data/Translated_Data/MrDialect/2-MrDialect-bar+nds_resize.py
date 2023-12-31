import csv
import sys
import time

with open('../Delimiters/Delimiter_NewLine.txt', 'r', encoding='utf-8') as file:
    delimiter_newline = file.read()

with open('../Delimiters/Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
    delimiter_tab = file.read()

dataset = [['bar_MrDialekt.tsv'], ['nds_MrDialekt.tsv'], ['de_Wiki_cleaned_resized_NewLine.tsv']]


def resize(target_size, files):
    filenames = []
    for file in files:
        filename = file[0]
        with open(filename, 'r', encoding='utf-16') as infile:
            reader = csv.reader(infile, delimiter=delimiter_newline)
            for row in reader:
                file.extend(row)

    for file in files:
        filename = file[0]
        data = file[1:]
        current_size = len(data)

        if current_size <= target_size:  # smallest file == target_size
            resized_data = data
        else:
            resized_data = data[:target_size]

        filepath = file[0]
        filename = str(str(target_size) + '/' + filepath[(filepath.rfind('/')):-4] + '_resized.tsv')
        filenames.append(filename)
        with open(filename, 'w', encoding='utf-16') as result_file:
            csv.writer(result_file, delimiter=delimiter_newline).writerow(resized_data)

    return filenames

#resize(target_size, dataset)