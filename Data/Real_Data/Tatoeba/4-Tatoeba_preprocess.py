import csv
import re
import unicodedata

csv.field_size_limit(262144)
with open('../Delimiters/Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
    delimiter_tab = file.read()

with open('../Delimiters/Delimiter_NewLine.txt', 'r', encoding='utf-8') as file:
    delimiter_newline = file.read()

filename = 'Tatoeba_de+nds+bar_mixed.tsv'


def remove_accent(text):  # remove accents
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return str(text)


def preprocess(filename, filename_output):

    with open(filename, 'r', encoding='utf-16') as infile:
        tsv_reader = csv.reader(infile, delimiter=delimiter_newline)
        for row in tsv_reader:
            for r in row:
                if r:
                    r = r.lower()  # string to lowercase
                    r = remove_accent(r)  # remove accented characters
                    r = re.sub(r'[^a-zA-ZäöüÄÖÜß \t]', '', r)  # remove punctuations, numbers and not relevant signs
                    r = re.sub(r'\b((?!de)\w{1,2})\b', '', r)  # remove short words or single letters
                    r = re.sub(r"[ \r\n]+", " ", r).strip()  # remove redundant white space
                    r = re.sub(r"(?<=\t|\n)\s+", "", r)  # remove extra white space after tab or new line
                    r = re.sub(r'\b(\w+)\b \1\b', r'\1', r)  # remove consecutive same words after preprocessing
                    if not re.fullmatch(r"[\s]+", r):  # remove empty rows exist only  Whitespaces = \s = (\r\t\n' ')
                        with open(filename_output, 'a', encoding='utf-16') as result_file:
                            result_file.write(r + delimiter_newline)


# preprocess(filename, str(filename[:-4] + '_preprocessed.tsv'))
