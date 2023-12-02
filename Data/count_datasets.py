import csv

with open('../Delimiters/Delimiter_NewLine.txt', 'r', encoding='utf-8') as file:
    delimiter_newline = file.read()

with open('../Delimiters/Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
    delimiter_tab = file.read()

files = [['filename.tsv'], ['filename2.tsv']]

for file in files:
    filename = file[0]
    tab_count = 0
    with open(filename, 'r', encoding='utf-16') as infile:
        for line in infile:
            tab_count += line.count(delimiter_tab)
        print(filename + ", Tab-Stopps " + str(tab_count + 1))
    with open(filename, 'r', encoding='utf-16') as infile:
        line_count = len(infile.readlines())
        print(filename + ", Lines: " + str(line_count))




