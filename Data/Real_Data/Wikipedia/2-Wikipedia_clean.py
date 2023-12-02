import csv
import time
import re

csv.field_size_limit(262144)
with open('Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
    delimiter = file.read()

filenames = ['data_de.tsv', 'data_nds.tsv', 'data_de.tsv']
t0 = time.time()
duplicates_list = []
j = 0

for fn in filenames:
    with open(fn, 'r', encoding='utf-16') as infile:
        tsv_reader = csv.reader(infile, delimiter=delimiter)
        for row in tsv_reader:
            if len(row) == 0:
                j = 0
            for r in row:
                if r.__contains__("https://bar.wikipedia.org/wiki/") or r.__contains__(" – Boarische Wikipedia") \
                        or r.__contains__("https://nds.wikipedia.org/wiki/") or r.__contains__(" – Wikipedia") \
                        or r.__contains__("https://de.wikipedia.org/wiki/"):
                    if r in duplicates_list:
                        j = 1
                    else:
                        duplicates_list.append(r)
                elif r.__contains__("→") \
                        or r.__contains__("↑") \
                        or r.__contains__("•") \
                        or r.__contains__("Liste von") \
                        or r.__contains__("Listn vo") \
                        or r.__contains__("List vun") \
                        or r.__contains__("ISBN") \
                        or r.__contains__("Der Artikl is im Dialekt") \
                        or r.__contains__("Dea Artikl is im Dialekt") \
                        or j == 1 \
                        or r == "" \
                        or r == delimiter \
                        or re.search(r'\d+\.\d*(\.\d+)?', r):  # contains enumerations
                    pass
                else:
                    r = re.sub(r'(?<!\s)\[.*?\](?!\s)', ' ', r)  # remove square brackets with content without space
                    r = re.sub(r'\[.*?\]', '', r)  # remove square brackets with content with space before or behind
                    r = re.sub(r'(?<!z\.[BT]\.)(?<=[a-z])(?=[A-Z])', ' ', r)  # split words at capitals, if low and then capital, but no point bevor capital
                    r = re.sub(r"\s+", " ", r).strip()  # remove extra white space left
                    r = re.sub(r'\b(\w+)\b \1\b', r'\1', r)  # remove consecutive same words after removing other things
                    with open(fn[:-4] + '_cleaned.tsv', 'a', encoding='utf-16') as result_file:
                        result_file.write(r + "\n")

    total_time = time.time() - t0
    with open('data.tsv', 'a', encoding='utf-16') as tsv_file:
        tsv_file.write("total time for clean up " + str(fn) + ": " + str(total_time) + " seconds!\n")
    print("total time for cleanup " + str(fn) + ": " + str(total_time) + " seconds!")
