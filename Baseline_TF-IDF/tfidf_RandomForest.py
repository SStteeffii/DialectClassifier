import numpy as np
import seaborn as sns
import matplotlib
from sklearn import svm, datasets
from joblib import dump, load
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from read_data import read_data, read_data_with_labels
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix, classification_report


def train_random_forest(filename, classifiername):

    X_train_tf, X_test_tf, training_label, test_label = load_split_vectorize_data(filename, 'train')

    # dimensionality reduction using PCA
    print('Dimensionality reduction...')
    pca = TruncatedSVD(n_components=128)
    x_train_tf_pca = pca.fit_transform(X_train_tf, None)
    print('Done!\n')

    # Classifier training RandomForest
    print('Classifier training...')
    classifier = RandomForestClassifier(verbose=1)
    classifier.fit(x_train_tf_pca, np.asarray(training_label))
    print('Done!\n')

    # save trained classifier
    print('Save Classifier...')
    dump(classifier, classifiername)
    print('Done!\n')

    return pca


def prediction_random_forest(classifiername, filename, data_label, pca):
    X_train_tf, X_test_tf, training_label, test_label = load_split_vectorize_data(filename, data_label)

    # load saved trained classifier
    print('Load Classifier...')
    model_loaded = load(classifiername)
    print('Done!\n')

    X_test_tf_pca = pca.transform(X_test_tf)

    # predict saved trained classifier with new test data
    print('Prediction...')
    y_pred = model_loaded.predict(X_test_tf_pca)
    print(classification_report(np.asarray(test_label), y_pred))
    with open('./Result_RandomForest/classification_report_RF_' + filename[17:filename.find("_", 18)] + '.tsv', 'a',
              encoding='utf-16') as outfile:
        outfile.write(str(classification_report(np.asarray(test_label), y_pred)))
    print('Done!\n')

    # save trained classifier
    print('Save Classifier...')
    clf = svm.SVC()
    X, y = datasets.load_iris(return_X_y=True)
    clf.fit(X, y)
    dump(clf, 'clf.joblib')
    print('Done!\n')

    plot_confusion_matrix(test_label, y_pred, data_label)


def load_split_vectorize_data(filename, data_label):
    print('Loading data...')
    splitlabels, datas, labels = read_data_with_labels(filename)
    labels = [0 if x == 'de' else x for x in labels]
    labels = [1 if x == 'bar' else x for x in labels]
    labels = [2 if x == 'nds' else x for x in labels]
    print('Done!\n')

    # split into training data validation data and test data
    print('Split data...')
    training_label, training_data, test_label, test_data, split_label = [], [], [], [], []
    for split_label, data, label in zip(splitlabels, datas, labels):
        if split_label == data_label:
            if data_label == 'validate' or data_label == 'test':
                test_data.append(data)
                test_label.append(label)
        if data_label == 'train':
            training_data.append(data)
            training_label.append(data)
    print('Done!\n')

    # vectorize the data with TF-IDF
    print('Vectorize data...')
    tf_idf = TfidfVectorizer()
    X_train_tf = tf_idf.fit_transform(training_data)
    X_test_tf = []
    if data_label == 'validate' or data_label == 'test':
        X_test_tf = tf_idf.transform(test_data)
    print('Done!\n')

    return X_train_tf, X_test_tf, training_label, test_label


def plot_confusion_matrix(labels_test, y_pred, data_label):
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
    plt.title('Confusion Matrix - TF-IDF, Random Forest', fontsize=18)
    plt.suptitle('Data: ' + file[17:file.find("_", 18)])

    plt.savefig('./Result_RandomForest/ConMat-TFIDF-RF-' + file[17:file.find("_", 18)] + "_" + data_label + '.png')
    print('Done!\n')


if __name__ == '__main__':

    files = ['../Data_Training/Dialektversum_de+nds+bar_mixed_preprocessed_splitlabel.tsv',
             '../Data_Training/MrDialect_de+nds+bar_mixed_preprocessed_splitlabel.tsv',
             '../Data_Training/Wikipedia_de+nds+bar_mixed_splitlabel.tsv']
    #files = ['../Data_Training/Wikipedia_de+nds+bar_mixed_splitlabel-half.tsv']
    #files = ['../Data_Training/Wikipedia_de+nds+bar_mixed_splitlabel-quater.tsv']

    file_final_test = '../Data_FinalTest/Tatoeba_de+nds+bar_mixed_preprocessed_splitlabel.tsv'

    classifier_name = 'clf.joblib'

    for file in files:

        # train classifier with train data
        pca = train_random_forest(file, classifier_name)

        # load trained classifier, test with validation data
        prediction_random_forest(classifier_name, file, 'validate', pca)

        # load trained classifier, test with test data
        prediction_random_forest(classifier_name, file, 'test', pca)

        # load trained classifier, test with final test data (Tatoeba)
        prediction_random_forest(classifier_name, file_final_test, 'test', pca)
