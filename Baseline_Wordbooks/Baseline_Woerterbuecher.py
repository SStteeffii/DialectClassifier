import csv
# import random
import time
import Bairisch
import Hochdeutsch
import Plattdeutsch
import Woerterbuch_result_evaluation

t0 = time.time()
count = 0

with open('../Delimiters/Delimiter_NewLine.txt', 'r', encoding='utf-8') as file:
    delimiter_newline = file.read()

with open('../Delimiters/Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
    delimiter_tab = file.read()

with open('../Data_FinalTest/Tatoeba_de+nds+bar_mixed_preprocessed.tsv', 'r', encoding='utf-16') as infile:
    tsv_reader = csv.reader(infile, delimiter=delimiter_newline)
    for row in tsv_reader:
        for r in row:
            try:
                bairisch_count = 0
                plattdeutsch_count = 0
                hochdeutsch_count = 0
                dialect, text_to_translate = r.split('\t', 1)
                words = text_to_translate.split()
                # if len(words) < 4:
                #     probability = 1  # each
                # elif len(words) < 8:
                #     probability = 0.5  # every 2nd
                # else:
                #     probability = 0.25  # every 4th
                for i in range(len(words)):
                    # if random.random() < probability:  # check random words in the text for dialect match
                    word_to_translate = words[i]
                    bairisch = Bairisch.BDO(word_to_translate) | Bairisch.BairischWoerterbuch(word_to_translate)
                    hochdeutsch = Hochdeutsch.Duden(word_to_translate) | Hochdeutsch.DWDS(word_to_translate)
                    plattdeutsch = Plattdeutsch.Platt_foer_Plietsche(word_to_translate) | Plattdeutsch.NDR(
                        word_to_translate) | Plattdeutsch.HillerPlatt(word_to_translate)
                    if bairisch:
                        bairisch_count += 1
                    if hochdeutsch:
                        hochdeutsch_count += 1
                    if plattdeutsch:
                        plattdeutsch_count += 1
                    time.sleep(1)
                dialects = {
                    "nds": plattdeutsch_count,
                    "bar": bairisch_count,
                    "de": hochdeutsch_count
                }
                if plattdeutsch_count == 0 and bairisch_count == 0 and hochdeutsch_count == 0:
                    highest_count_dialect = "none"
                else:
                    max_count_value = max(dialects.values())
                    highest_count_dialects = [dialect for dialect, count in dialects.items() if
                                              count == max_count_value]
                    highest_count_dialect = ", ".join(highest_count_dialects)

                if dialect == highest_count_dialect:
                    right_dialect = True
                else:
                    right_dialect = False

                with open('./Result/data_baseline_result.tsv', 'a', encoding='utf-16') as tsv_file:
                    tsv_file.write(
                        r + delimiter_tab + highest_count_dialect + delimiter_tab + str(right_dialect) + "\n")

            except Exception as e:
                with open('./Result/error_data_baseline.tsv', 'a', encoding='utf-16') as outfile:
                    outfile.write(r)
                    outfile.write(delimiter_newline)
                with open('./Result/errors_baseline.tsv', 'a', encoding='utf-16') as outfile:
                    outfile.write(str(e))
                    outfile.write(delimiter_newline)

            finally:
                time.sleep(1)
                if count % 100 == 0 and count != 0:
                    Woerterbuch_result_evaluation.result_evaluation()
                count += 1

total_time = time.time() - t0
with open('./Result/data.tsv', 'a', encoding='utf-16') as tsv_file:
    tsv_file.write("total time = " + str(total_time) + " seconds!\n")
print("total time = " + str(total_time) + " seconds!")
