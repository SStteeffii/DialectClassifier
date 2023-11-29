import csv
import re
import requests
import xml.etree.ElementTree as ET


def BDO(word):
    url = "https://bdo.badw.de/api/v1"

    params = {
        # "stichwort": word, #zu allgemein
        "lemma": word,  # nur lemma
        "case": "no",
        "exact": "yes",
        "compress": "no"
    }
    headers = {
        "Accept": "text/xml,text/html,application/xml",
        "Cache-Control": "max-age=0,no-cache,no-store,must-revalidate",
        "Pragma": "no-cache",
        "expires": "0"
    }

    response = requests.get(url, params=params, headers=headers, timeout=20)
    if response.status_code == 200:
        root = ET.fromstring(response.text)
        result_count = root.find(".//result_count").text
        if int(result_count) == 0:
            return False
        else:
            return True


# Quelle: https://deutsch-bairisch.de/vokabeln/ - Copy-and-Paste in eine CSV-Datei
# Quelle: https://www.traunsteiner-tagblatt.de/bayrisch-woerterbuch.html - Copy-and-Paste in eine CSV-Datei
def BairischWoerterbuch(word):
    pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
    with open("Bairisch Wörterbuch.csv", 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')

        # Überspringt die Kopfzeile
        next(reader, None)

        for row in reader:
            # Prüfen, ob das Wort in einer der Spalten B, C oder D vorkommt
            if pattern.search(row[1]) or pattern.search(row[2]) or pattern.search(row[3]):
                return True
        return False


# print(BDO("servus"))
# print(BairischWoerterbuch("servus"))
