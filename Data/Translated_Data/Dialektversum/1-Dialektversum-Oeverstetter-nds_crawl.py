import time
from lxml import html
import requests
import csv

t0 = time.time()
url = "https://www.xn--versetter-z7a.de/translation"

with open('data_de_cleaned_resized_V13.tsv', 'r', encoding='utf-16') as infile:
    tsv_reader = csv.reader(infile, delimiter='\t')
    for row in tsv_reader:
        for r in row:

            text_to_translate = r

            try:
                params = {"jsdata": text_to_translate, "lang": "nds"}
                response = requests.get(url, params=params, timeout=20)
                if response.status_code == 200:
                    tree = html.fromstring(response.content.decode('utf-8'))
                    translated_text = tree.xpath('//input[@name="translation"]/@value')[0]
                else:
                    with open('output_error_Dialektversum-Oeverstetter_plattdeutsch.tsv', 'a',
                              encoding='utf-16') as outfile:
                        outfile.write(text_to_translate)
                        outfile.write("\t")

                if translated_text:
                    with open('nds_Dialektversum-Oeverstetter.tsv', 'a', encoding='utf-16') as outfile:
                        outfile.write(translated_text)
                        outfile.write("\t")
            except Exception:
                with open('output_error_Dialektversum-Oeverstetter_plattdeutsch.tsv', 'a', encoding='utf-16') as outfile:
                    outfile.write(text_to_translate)
                    outfile.write("\t")
            finally:
                time.sleep(0.6)

total_time = time.time() - t0
with open('data.tsv', 'a', encoding='utf-16') as tsv_file:
    tsv_file.write("total time plattdeutsch = " + str(total_time) + " seconds!\n")
print("total time = " + str(total_time) + " seconds!")
