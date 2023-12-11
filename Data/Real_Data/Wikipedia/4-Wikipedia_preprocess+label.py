import csv
import time
import re
import unicodedata

csv.field_size_limit(262144)
with open('../Delimiters/Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
    delimiter_tab = file.read()

with open('../Delimiters/Delimiter_NewLine.txt', 'r', encoding='utf-8') as file:
    delimiter_newline = file.read()

filenames = ['data_bar_cleaned_resized.tsv', 'data_nds_cleaned_resized.tsv', 'data_de_cleaned_resized.tsv']


def remove_accent(text):
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return str(text)


def preprocess(filename):
    filenames_output = []
    for fn in filename:
        dialect_name = fn.split("_")[1]
        filename_output = str(fn[:-4] + '_preprocessed.tsv')
        filenames_output.append(filename_output)
        with open(fn, 'r', encoding='utf-16') as infile:
            tsv_reader = csv.reader(infile, delimiter=delimiter_newline)
            for row in tsv_reader:
                for r in row:
                    if r:
                        r = r.lower()  # string to lowercase
                        r = remove_accent(r)  # remove accented characters
                        r = re.sub(r'[^a-zA-ZäöüÄÖÜß ]', '', r)  # remove punctuations, numbers and not relevant signs
                        r = re.sub(r'\b(\w{1,2})\b', '', r)  # remove short words or single letters, they're mostly stopwords
                        r = re.sub(r"\s+", " ", r).strip()  # remove extra white space left
                        r = re.sub(r'\b(\w+)\b \1\b', r'\1', r)  # remove consecutive same words after preprocessing
                        r = re.sub(r"(?<=\t|\n)\s+", "", r)  # remove extra white space after tab or new line
                        if not re.fullmatch(r"[\s]+", r):  # remove empty rows exist only  Whitespaces = \s = (\r\t\n' ')
                            with open(filename_output, 'a', encoding='utf-16') as result_file:
                                result_file.write(dialect_name + delimiter_tab + r + delimiter_newline)

    return filenames_output

# preprocess(filename, str(filename[:-4] + '_preprocessed.tsv'))
