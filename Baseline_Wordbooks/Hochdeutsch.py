import csv
import json

import duden
import simplemma


def Duden(word):
    try:
        lemma = simplemma.lemmatize(word, lang='de')
        found = duden.search(lemma)  # nur lemma
        if found:
            return True
        else:
            return False
    except Exception as e:
        print(str(e))


def DWDS(word):
    try:
        lemma = simplemma.lemmatize(word, lang='de').lower()

        with open("dwds_lemmata_2023-11-02.csv", 'r', encoding='utf-8') as file:  # DWDS-Wörterbuch (Lemmadatenbank) https://www.dwds.de/lemma/list https://www.dwds.de/lemma/csv
            reader = csv.reader(file, delimiter=',')

            # Überspringt die Kopfzeile
            next(reader, None)

            for row in reader:
                if row:
                    # Prüfen, ob das Wort in der ersten Spalte vorkommt
                    first_column = row[0].lower()
                    for word in first_column:
                        if word == lemma:
                            return True
        with open("dwds_wdg-headwords.json", 'r', encoding='utf-8') as file2:  # DWDS-Wörterbuch https://www.dwds.de/dwds_static/wb/wdg-headwords.json
            data = json.load(file2)

            if lemma in (key.lower() for key in data.keys()):
                return True

        with open("dwds_dwdswb-headwords.json", 'r', encoding='utf-8') as file3:  # Wörterbuch der deutschen Gegenwartssprache https://www.dwds.de/dwds_static/wb/dwdswb-headwords.json
            data = json.load(file3)

            if lemma in (key.lower() for key in data.keys()):
                return True
        return False
    except Exception as e:
        print(str(e))


# print(Duden("servus"))
# print(DWDS("faxanschluß"))
