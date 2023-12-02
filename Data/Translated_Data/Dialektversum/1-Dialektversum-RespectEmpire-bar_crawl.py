import csv
import time

from lxml import html
import requests
import re

t0 = time.time()
url = "https://www.respekt-empire.de/Translator/?page=translateEngine"

with open('data_de_cleaned_resized.tsv', 'r', encoding='utf-8') as infile:
    tsv_reader = csv.reader(infile, delimiter='\t')
    for row in tsv_reader:
        for r in row:
            text_to_translate = r
            try:
                session = requests.Session()

                if len(text_to_translate) > 700:  # otherwise timeout
                    text_to_translate = text_to_translate[:text_to_translate.find(".", 300)+1]  # split into full sentence

                text_to_translate = text_to_translate.replace("é", "e")

                response = session.get(url)
                if response.status_code == 200:
                    form_data = {
                        'tr_text': text_to_translate,
                        'translate': 'Übersetzen'
                    }
                    x = (len(text_to_translate))
                    response = session.post(url, data=form_data)
                    if response.status_code == 200:
                        tree = html.fromstring(response.content)
                        translated_text = tree.xpath('//div[@class="translator"]/text()')[0]
                    else:
                        break

                    if translated_text:
                        translated_text = re.sub(r'\s+', ' ', translated_text).strip()
                        with open('bar_Dialektversum-RespektEmpire.tsv', 'a', encoding='utf-8') as outfile:
                            outfile.write(translated_text)
                            outfile.write("\t")
                else:
                    break
            except IndexError or ConnectionError:
                with open('output_indexORconnectionerror_Dialektversum-RespektEmpire_bairisch.tsv', 'a', encoding='utf-8') as outfile:
                    outfile.write(text_to_translate)
                    outfile.write("\t")
            except Exception:
                with open('output_othererror_Dialektversum-RespektEmpire_bairisch.tsv', 'a', encoding='utf-8') as outfile:
                    outfile.write(text_to_translate)
                    outfile.write("\t")
            finally:
                time.sleep(0.5)

total_time = time.time() - t0
with open('data.tsv', 'a', encoding='utf-16') as tsv_file:
    tsv_file.write("total time bairisch with error detection = " + str(total_time) + " seconds!\n")
print("total time = " + str(total_time) + " seconds!")
