import json
import random
import time
import requests
import csv

t0 = time.time()
url = "https://translator-ai.onrender.com/"
headers = {
    "Content-Type": "application/json",
    #  "Referer": "https://de.mr-dialect.com/"  # spoofing - pretend to be someone else -> request is working with this line
}
dialects = ["bairisch", "plattdeutsch"]

with open('Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
    delimiter_tab = file.read()

with open('data_de_cleaned_resized.tsv', 'r', encoding='utf-16') as infile:
    tsv_reader = csv.reader(infile, delimiter=delimiter_tab)
    for row in tsv_reader:
        for r in row:
            if len(r) > 1000:
                text_to_translate = r[:r.find(".", 800)+1]
            else:
                text_to_translate = r
            try:
                for d in dialects:
                    data = {
                        "prompt": f'Übersetze den Hochdeutschen Text "{text_to_translate}", in den Dialekt {d}.'
                    }
                    response = requests.post(url, json=data, headers=headers)
                    if response.status_code == 200:
                        response_content = json.loads(response.content.decode('utf-8'))
                        translated_text = response_content.get('bot', None)
                    else:
                        break

                    if translated_text:
                        with open(d + '_MrDialekt' + '.tsv', 'a', encoding='utf-16') as outfile:
                            outfile.write(translated_text)
                            outfile.write(delimiter_tab)
            except ConnectionError:
                with open('output_connectionerror_MrDialekt_' + d + '.tsv', 'a', encoding='utf-16') as outfile:
                    outfile.write(text_to_translate)
                    outfile.write(delimiter_tab)
            except Exception:
                with open('output_othererror_MrDialekt_' + d + '.tsv', 'a', encoding='utf-16') as outfile:
                    outfile.write(text_to_translate)
                    outfile.write(delimiter_tab)
            finally:
                time.sleep(random.uniform(0.30, 0.80))

total_time = time.time() - t0
with open('data.tsv', 'a', encoding='utf-16') as tsv_file:
    tsv_file.write("total time= " + str(total_time) + " seconds!\n")
print("total time = " + str(total_time) + " seconds!")
