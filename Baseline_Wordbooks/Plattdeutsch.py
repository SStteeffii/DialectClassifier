import csv

import requests
from bs4 import BeautifulSoup
import re


def Platt_foer_Plietsche(word):
    url = "https://www.plattdeutsches-woerterbuch.de/search"

    data = {
        "searchterm": word
    }

    response = requests.post(url, data=data, timeout=20)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'lxml')

        p_element = soup.find("p", class_="platthead")
        if p_element and "Ihr Suchergebnis:" in p_element.get_text():  # found Results in Plattdeutsch

            if p_element:
                # Search in Table after "Ihr Suchergebnis"
                table = p_element.find_next("table", class_="table table-bordered resultTable")
                if table:
                    # Durchsuchen Sie alle <td> Zellen der ersten Spalte
                    for row in table.find_all("tr"):
                        td_cells = row.find_all("td")
                        if td_cells and td_cells[
                            0].text.strip() == word:  # Plattdeutsch is always in the first column
                            return True
            return False


# print(Platt_för_Plietsche("Moin"))

def NDR(word):
    url = "https://www.ndr.de/kultur/norddeutsche_sprache/plattdeutsch/woerterbuch101.jsp?"

    param = {
        "suchbegriff": word
    }
    headers = {
        "Accept": "text/xml,text/html,application/xml",
        "Cache-Control": "max-age=0,no-cache,no-store,must-revalidate",
        "Pragma": "no-cache",
        "expires": "0"
    }

    response = requests.get(url, params=param, timeout=20)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'lxml')
        table = soup.find("table", summary="Platdeutsch Wörterbuch")

        if table and table.find("caption").get_text() == "Suchergebnis":
            for row in table.find_all("tr"):
                td = row.find("td", headers="Woort")
                if td:
                    words_in_td = re.findall(r'\b{}\b'.format(re.escape(word)), td.get_text(), re.IGNORECASE)
                    if words_in_td:
                        return True
    return False


# Quelle: https://github.com/hillerplatt/hillerplatt.de
def HillerPlatt(word):
    pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)

    with open("wortliste_hillerplatt.csv", 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',')

        # Überspringt die Kopfzeile
        next(reader, None)

        for row in reader:
            # Prüfen, ob das Wort in einer der Spalten B, C oder D vorkommt
            if pattern.search(row[1]):
                return True
        return False

# print(HillerPlatt("Zoahl"))
