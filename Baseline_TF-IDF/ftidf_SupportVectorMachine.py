from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from read_data import read_data, read_data_with_labels
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn import svm, datasets
from joblib import dump, load
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import random
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_split_vectorize_data(filename, data_label, tf_idf=None):
    print('Loading data...')
    splitlabels, datas, labels = read_data_with_labels(filename)
    labels = [0 if x == 'de' else x for x in labels]
    labels = [1 if x == 'bar' else x for x in labels]
    labels = [2 if x == 'nds' else x for x in labels]
    print('Done!\n')

    # split into training data validation data and test data
    print('Split data...')
    label_list, data_list, split_label = [], [], []
    for split_label, data, label in zip(splitlabels, datas, labels):
        if split_label == data_label:
            data_list.append(data)
            label_list.append(label)
    print('Done!\n')

    # vectorize the data with TF-IDF
    print('Vectorize data...')

    if tf_idf is None:
        tf_idf = TfidfVectorizer()

    if data_label == 'train':
        x_tf = tf_idf.fit_transform(data_list)
    elif data_label == 'validate' or data_label == 'test':
        x_tf = tf_idf.transform(data_list)
    else:
        x_tf = None
    print('Done!\n')

    return x_tf, label_list, tf_idf


def train_svm(filename, classifiername, num_samples: int):
    # load training data, corrresponding labels as well as the tf_idf vectorizer
    x_train_tf, training_label, tf_idf = load_split_vectorize_data(filename, 'train')

    # dimensionality reduction using PCA
    print('Dimensionality reduction...')
    pca = TruncatedSVD(n_components=128)
    x_train_tf_pca = pca.fit_transform(x_train_tf, None)
    print('Done!\n')

    # sample indices which correspond to the training data
    indices = random.choices(list(range(len(training_label))), k=num_samples)

    # store the corresponding data and labels
    x_train_tf_pca = np.take(x_train_tf_pca, indices, axis=0)
    training_label = [training_label[idx] for idx in indices]

    # Classifier training RandomForest
    print('Classifier training...')
    classifier = LinearSVC()
    classifier.fit(x_train_tf_pca, np.asarray(training_label), )
    print('Done!\n')

    # save trained classifier
    print('Save Classifier...')
    dump(classifier, classifiername)
    print('Done!\n')
    return pca, tf_idf


def prediction_svm(classifiername, output_path, filename_out, filename, data_label, pca, tf_idf):
    # load the test data
    x_test_tf, test_label, _ = load_split_vectorize_data(filename, data_label, tf_idf)

    # load saved trained classifier
    print('Load Classifier...')
    model_loaded = load(classifiername)
    print('Done!\n')

    # perform a dimensionality reduction
    x_test_tf_pca = pca.transform(x_test_tf)

    # predict saved trained classifier with new test data
    print('Prediction...')
    y_pred = model_loaded.predict(x_test_tf_pca)
    # print(classification_report(np.asarray(test_label), y_pred))

    # create the directory in which the results are stored
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # store the results
    with open(output_path + filename_out + '.tsv', 'w', encoding='utf-16') as outfile:
        outfile.write(str(classification_report(np.asarray(test_label), y_pred)))
    print('Done!\n')

    plot_confusion_matrix(test_label, y_pred, output_path, filename_out)


def plot_confusion_matrix(labels_test, y_pred, output_path, filename_out):
    # Plot confusion matrix in a beautiful manner
    print('Plot Confustion Matrix...')
    ax = plt.subplot()
    index = ['Hochdeutsch', 'Bairisch', 'Plattdeutsch']

    cnf_matrix = confusion_matrix(labels_test, y_pred)
    df = pd.DataFrame(cnf_matrix, index=index, columns=index)
    sns.heatmap(df, annot=True, cmap='Blues', ax=ax, fmt='g')

    # labels, title and ticks
    ax.set_xlabel('Predicted', fontsize=15)
    ax.set_ylabel('True', fontsize=15)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(output_path + filename_out + '.png')
    print('Done!\n')


if __name__ == '__main__':

    # files = [# '../Data_Training/Dialektversum_de+nds+bar_mixed_preprocessed_splitlabel.tsv',
    #         # '../Data_Training/MrDialect_de+nds+bar_mixed_preprocessed_splitlabel.tsv',
    #         '../Data_Training/Wikipedia_de+nds+bar_mixed_splitlabel.tsv']
    # files = ['../Data_Training/Wikipedia_de+nds+bar_mixed_splitlabel-half.tsv']

    # files = ['../Data_Training/Wikipedia_de+nds+bar_mixed_splitlabel-half.tsv']

    files_train = ['../Data_Training/MrDialect_de+nds+bar_mixed_preprocessed_splitlabel.tsv']
    file_validation = '../Data_Training/Wikipedia_de+nds+bar_mixed_splitlabel-half.tsv'
    file_final_test = '../Data_FinalTest/Tatoeba_de+nds+bar_mixed_preprocessed_splitlabel.tsv'

    output_path = 'Result_SupportVectorMachine/mr_dialect/run_5/'
    classifier_name = 'svm.joblib'
    num_samples_list = [20, 40, 60, 80, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]

    for num_samples in num_samples_list:
        output_path_act = output_path + str(num_samples) + '/'
        print(num_samples)
        for file_train in files_train:
            # train classifier with train data
            pca, tf_idf = train_svm(file_train, classifier_name, num_samples)

            # load trained classifier, test with validation data
            file_name_val = 'wikipedia_validation'
            prediction_svm(classifier_name, output_path_act, file_name_val, file_validation, 'validate', pca, tf_idf)

            # load trained classifier, test with test data
            file_name_test = 'wikipedia_test'
            prediction_svm(classifier_name, output_path_act, file_name_test, file_validation, 'test', pca, tf_idf)

            # load trained classifier, test with final test data (Tatoeba)
            file_name_final = 'tatoeba'
            prediction_svm(classifier_name, output_path_act, file_name_final, file_final_test, 'test', pca, tf_idf)
