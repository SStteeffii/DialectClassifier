import csv
import time

with open('Delimiter_NewLine.txt', 'r', encoding='utf-8') as file:
    delimiter_newline = file.read()

with open('Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
    delimiter_tab = file.read()

files = [['data_bar_cleaned.tsv'], ['data_nds_cleaned.tsv'], ['data_de_cleaned.tsv']]

for file in files:
    filename = file[0]
    with open(filename, 'r', encoding='utf-16') as infile:
        reader = csv.reader(infile, delimiter=delimiter_tab)
        for row in reader:
            file.extend(row)

# Determine the target size as the size of the smallest file
target_size = 36256  # size of the smallest data set

for file in files:
    t0 = time.time()
    filename = file[0]
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

    with open(filename[:-4] + '_resized.tsv', 'w', encoding='utf-16') as result_file:
        csv.writer(result_file, delimiter=delimiter_newline).writerow(resized_data)

    total_time = time.time() - t0
    with open('data_resizing.tsv', 'a', encoding='utf-16') as tsv_file:
        tsv_file.write("total time for resizing " + str(filename) + ": " + str(total_time) + " seconds!\n")
    print("total time for resizing " + str(filename) + ": " + str(total_time) + " seconds!")
