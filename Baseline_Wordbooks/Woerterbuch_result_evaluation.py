import csv
import numpy as np
import seaborn as sns
import matplotlib
from sklearn.metrics import confusion_matrix, classification_report
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def result_evaluation():
    with open('../Delimiters/Delimiter_NewLine.txt', 'r', encoding='utf-8') as file:
        delimiter_newline = file.read()

    with open('../Delimiters/Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
        delimiter_tab = file.read()

    true_count = 0
    false_count = 0
    none_count = 0
    count_rows = 0
    labels_test = []
    labels_predict = []

    with open('./Result/data_baseline_result.tsv', 'r', encoding='utf-16') as infile:
        tsv_reader = csv.reader(infile, delimiter=delimiter_newline)
        for row in tsv_reader:
            count_rows += 1
            for r in row:
                splitlabel, dialect, text_to_translate, woerterbuch_result, match = r.split(delimiter_tab)
                if match == "True":
                    true_count += 1
                if match == "False":
                    false_count += 1
                if woerterbuch_result == "none":
                    none_count += 1
                if woerterbuch_result == "de" or woerterbuch_result == "nds" or woerterbuch_result == "bar":
                    labels_test.append(dialect)
                    labels_predict.append(woerterbuch_result)
    mapping = {'de': 0, 'bar': 1, 'nds': 2}
    labels_test = [mapping.get(x, x) for x in labels_test]
    labels_predict = [mapping.get(x, x) for x in labels_predict]

    print((true_count / count_rows) * 100)
    print(classification_report(np.asarray(labels_test), labels_predict))

    # Plot confusion matrix
    ax = plt.subplot()
    index = ['Hochdeutsch', 'Bairisch', 'Plattdeutsch']

    cnf_matrix = confusion_matrix(labels_test, labels_predict)
    np.set_printoptions(precision=2, suppress=True, formatter={'float': '{:0.0f}'.format})
    df = pd.DataFrame(cnf_matrix, index=index, columns=index)
    sns.heatmap(df, annot=True, cmap='Blues', ax=ax, fmt='.2g')

    # labels, title and ticks
    ax.set_xlabel('Predicted', fontsize=15)
    ax.set_ylabel('True', fontsize=15)
    plt.title('Confusion Matrix - Wörterbuch Tatoeba', fontsize=18)

    filename_conmat = "./Result/ConfustionMatrix-Woerterbuecher-Tatoeba-" + str(count_rows) + ".png"
    plt.savefig(str(filename_conmat))

    with open('./Result/baseline_result_evaluation.tsv', 'a', encoding='utf-16') as outfile:
        outfile.write(str(round((true_count / count_rows) * 100, 2)) + "%, total rows: " + str(
            count_rows) + ", false and none rows: " + str(none_count) + ", false rows: " + str(
            false_count) + ", true rows: " + str(true_count))
        outfile.write(delimiter_newline)
        outfile.write(str(classification_report(np.asarray(labels_test), labels_predict)))
        outfile.write(delimiter_newline)
