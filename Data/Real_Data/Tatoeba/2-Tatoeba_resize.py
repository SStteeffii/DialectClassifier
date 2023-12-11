import csv

with open('../Delimiters/Delimiter_NewLine.txt', 'r', encoding='utf-8') as file:
    delimiter_newline = file.read()

with open('../Delimiters/Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
    delimiter_tab = file.read()

dataset = [['bar_Tatoeba_labeled.tsv'], ['de_Tatoeba_labeled.tsv'], ['nds_Tatoeba_labeled.tsv']]


def resize(target_size, files):
    filenames = []
    for file in files:
        with open(file[0], 'r', encoding='utf-16') as infile:
            reader = csv.reader(infile, delimiter=delimiter_newline)
            for row in reader:
                 file.extend(row)

    for file in files:
        data = file[1:]
        current_size = len(data)

        if current_size <= target_size:  # smallest file == target_size
            resized_data = data
        else:
            resized_data = []
            step_size = current_size / target_size
            i = 0
            current_index = 0
            for row in data:
                if i <= current_index:
                    resized_data.append(row)
                    i += step_size
                current_index += 1
        filepath = file[0]
        filename = str(str(target_size) + '/' + filepath[(filepath.rfind('/')):-4] + '_resized.tsv')
        filenames.append(filename)
        with open(filename, 'w', encoding='utf-16') as result_file:
            csv.writer(result_file, delimiter=delimiter_newline).writerow(resized_data)
    return filenames


#for file in data:
#    filename = file[0]
#       with open(filename, 'r', encoding='utf-16') as infile:
#        reader = csv.reader(infile, delimiter=delimiter_newline)
#        for row in reader:
#            file.extend(row)
#
# Determine the target size as the size of the smallest file
#target_size = min(len(data[0]), len(data[1]), len(data[2]))
#
#resize(target_size, dataset)
