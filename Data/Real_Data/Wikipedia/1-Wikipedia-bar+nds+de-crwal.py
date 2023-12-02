import re
import time
from bs4 import BeautifulSoup
import requests

t0 = time.time()  # for tracking duration, starttime
tnext = time.time()
tend = 0
start_links = ["/wiki/Deitsche_Sproch", "/wiki/Düütsche_Spraak", "/wiki/Deutsche_Sprache"]  # start links for languages
links = []  # collect links to iterate it
languages = ["bar", "nds", "de"]  # bar = bairisch, nds = plattdetusch(niederdeutsch), de = deutsch
languages_start_links = dict(zip(languages, start_links))  # dictionary with language and start link for the language
all_used_links = []  # visited links - should only visit once
forbidden_link_starts = ["/wiki/Wikipedia", "/wiki/Spezial", "/wiki/Special", "/wiki/Datei", "/wiki/File",
                         "/wiki/Kategorie", "/wiki/Portal", "/wiki/Liste", "/wiki/Vorlog", "/wiki/User", "/wiki/Nutza",
                         "/wiki/Dischkrian", "/wiki/Diskuschoon", "/wiki/Dischkrian", "/wiki/Dischkrian", "/wiki/Buach",
                         "/wiki/Buch", "/wiki/Diskussion", "/wiki/Benutzer", "/wiki/ISO", "/wiki/Bild"]
# Wikipedia links are not visited
forbidden_contents = ["[Werkeln | Am Gwëntext werkeln]", "[ännern | Bornkood ännern]",
                      "[Bearbeiten | Quelltext bearbeiten]", "Koordinaten"]  # data cleaned from forbidden contents
stop_contents = ["Beleg", "Biacha", "Im Netz", "Weblinks", "Enkeld Nahwiesen", "Literatur", "Literatua", "Weblenken",
                 "Footnoten", "Einzelnachweise", "Lenken"]  # the parsing ends at this key word
i0 = 0

for lan in languages:
    link = languages_start_links.pop(lan, None)
    links.append(link)
    tnext = time.time()
    i = 0
    while links:
        link = links.pop()
        all_used_links.append(link)
        clean_content = []
        clean_long_content = []
        try:
            response = requests.get("https://" + lan + ".wikipedia.org" + link, timeout=10)

            soup = BeautifulSoup(response.text, 'lxml')  # lxml or html.parser

            output_title = "\n" + soup.title.string
            output_link = "https://" + lan + ".wikipedia.org" + link

            content = soup.find_all(class_='mw-parser-output')
            for item in content:
                clean_content.append(item.get_text())
            clean_content = [part for string in clean_content for part in string.split("\n")]

            for index, clean_item in enumerate(clean_content):
                if any(c in clean_item for c in stop_contents) and index >= len(clean_content) * 1 / 2:
                    break
                if len(clean_item) > 40 and not any(s in clean_item for s in forbidden_contents):
                    clean_long_content.append(clean_item)

            with open('data_' + lan + '.tsv', 'a', encoding='utf-16') as tsv_file:
                tsv_file.write(output_title)
                tsv_file.write("\t")
                tsv_file.write(output_link)
                tsv_file.write("\t")
                tsv_file.write("\t".join(clean_long_content))
                tsv_file.write("\n")

            # save all linked pages from the current page
            for links_on_page in soup.find_all('a'):
                link_name = links_on_page.get('href')

                if str(link_name).startswith("/wiki/") \
                        and not re.match(r'/wiki/\d{1,4}', str(link_name)) \
                        and not any(str(link_name).startswith(fls) for fls in forbidden_link_starts) \
                        and not str(link_name) in all_used_links \
                        and not str(link_name) in links:
                    links.append(link_name)

            i = i + 1
            time.sleep(0.2)
        except Exception as err:
            print(link)
            print(links)
            print(all_used_links)
            print(err)

    i0 = i0 + i
    tend = time.time()
    with open('data_' + lan + '.tsv', 'a', encoding='utf-16') as tsv_file:
        tsv_file.write("total time = " + str(tend - tnext) + " seconds")
        tsv_file.write("Count Wiki Sites  = " + str(i))

    with open('data.tsv', 'a', encoding='utf-16') as tsv_file:
        tsv_file.write("total time for language " + lan + "= " + str(tend - tnext) + " seconds!\n")
        tsv_file.write("Count Wiki Sites for language " + lan + "= " + str(i) + "\n")

    print("total time for language " + lan + "= " + str(tend - tnext) + " seconds with " + str(i) + " iterations!\n")

t1 = time.time()
total_time = tend - t0
with open('data.tsv', 'a', encoding='utf-16') as tsv_file:
    tsv_file.write("total time = " + str(total_time) + " seconds!\n")
    tsv_file.write("total iterations = " + str(i0) + " times!\n")
print("total time = " + str(total_time) + " seconds!")
