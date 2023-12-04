import numpy as np
import seaborn as sns
import matplotlib
from joblib import dump, load
from sklearn import svm, datasets

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from read_data import read_data, read_data_with_labels
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix, classification_report


if __name__ == '__main__':

    files = ['../Data_Training/Dialektversum_de+nds+bar_mixed_preprocessed_splitlabel.tsv',
             '../Data_Training/MrDialect_de+nds+bar_mixed_preprocessed_splitlabel.tsv',
             '../Data_Training/Wikipedia_de+nds+bar_mixed_splitlabel.tsv']

    file_final_test = '../Data_FinalTest/Tatoeba_de+nds+bar_mixed_preprocessed_splitlabel.tsv'

    classifier_name = 'clf.joblib'

    for file in files:

        classifier_name = 'clf.joblib'

        print('Loading data...')
        splitlabels, datas, labels = read_data_with_labels(file)
        labels = [0 if x == 'de' else x for x in labels]
        labels = [1 if x == 'bar' else x for x in labels]
        labels = [2 if x == 'nds' else x for x in labels]
        print('Done!\n')

        # split into training data validation data and test data
        data_train, labels_train, data_validation, labels_validation, data_test, labels_test = [], [], [], [], [], []
        for splitlabel, data, label in zip(splitlabels, datas, labels):
            if splitlabel == 'train':
                data_train.append(data)
                labels_train.append(label)
            elif splitlabel == 'validate':
                data_validation.append(data)
                labels_validation.append(label)
            elif splitlabel == 'test':
                data_test.append(data)
                labels_test.append(label)

        # vectorize the data
        print('Vectorize data...')
        tf_idf = TfidfVectorizer()
        X_train_tf = tf_idf.fit_transform(data_train)
        print('Done!\n')

        # dimensionality reduction using PCA
        #print('Dimensionality reduction...')
        #pca = TruncatedSVD(n_components=128)
        #X_train_tf_pca = pca.fit_transform(X_train_tf, None)
        #print('Done!\n')

        # Classifier training Gradient Boosting Classifier
        print('Classifier training...')
        classifier = LinearSVC()
        classifier.fit(X_train_tf, np.asarray(labels_train))
        print('Done!\n')

        # save trained classifier
        print('Save Classifier...')
        dump(classifier, classifier_name)
        print('Done!\n')

        # load saved trained classifier
        print('Load Classifier...')
        model_loaded = load(classifier_name)
        print('Done!\n')

        X_validate_tf = tf_idf.transform(data_validation)

        # predict test data
        y_pred = model_loaded.predict(X_validate_tf)
        print(classification_report(np.asarray(labels_validation), y_pred))
        with open('./Result_SupportVectorMachine/classification_report_SVM_val-' + file[17:file.find("_", 18)] + '.tsv', 'a', encoding='utf-16') as outfile:
            outfile.write(str(classification_report(np.asarray(labels_validation), y_pred)))

        # save trained classifier
        print('Save Classifier...')
        clf = svm.SVC()
        X, y = datasets.load_iris(return_X_y=True)
        clf.fit(X, y)
        dump(clf, classifier_name)
        print('Done!\n')

        # Plot confusion matrix in a beautiful manner
        ax = plt.subplot()
        index = ['Hochdeutsch', 'Bairisch', 'Plattdeutsch']

        cnf_matrix = confusion_matrix(labels_validation, y_pred)
        df = pd.DataFrame(cnf_matrix, index=index, columns=index)
        sns.heatmap(df, annot=True, cmap='Blues', ax=ax)  # fmt='g'

        # labels, title and ticks
        ax.set_xlabel('Predicted', fontsize=15)
        ax.set_ylabel('True', fontsize=15)
        plt.title('Refined Confusion Matrix - TF-IDF, Support Vector Machine', fontsize=18)
        plt.suptitle('Data: ' + file[17:file.find("_", 18)])

        plt.savefig('./Result_SupportVectorMachine/ConMat-TFIDF-SVM-val-' + file[17:file.find("_", 18)] + '.png')

        # load saved trained classifier
        print('Load Classifier...')
        model_loaded = load(classifier_name)
        print('Done!\n')

        X_train_tf = tf_idf.fit_transform(data_train)
        X_test_tf = tf_idf.transform(data_test)

        # predict test data
        y_pred = model_loaded.predict(X_test_tf)
        print(classification_report(np.asarray(labels_test), y_pred))
        with open('./Result_SupportVectorMachine/classification_report_SVM_test1-' + file[17:file.find("_", 18)] + '.tsv', 'a', encoding='utf-16') as outfile:
            outfile.write(str(classification_report(np.asarray(labels_test), y_pred)))

        # save trained classifier
        print('Save Classifier...')
        clf = svm.SVC()
        X, y = datasets.load_iris(return_X_y=True)
        clf.fit(X, y)
        dump(clf, classifier_name)
        print('Done!\n')

        # Plot confusion matrix in a beautiful manner
        ax = plt.subplot()
        index = ['Hochdeutsch', 'Bairisch', 'Plattdeutsch']

        cnf_matrix = confusion_matrix(labels_validation, y_pred)
        df = pd.DataFrame(cnf_matrix, index=index, columns=index)
        sns.heatmap(df, annot=True, cmap='Blues', ax=ax)  # fmt='g'

        # labels, title and ticks
        ax.set_xlabel('Predicted', fontsize=15)
        ax.set_ylabel('True', fontsize=15)
        plt.title('Refined Confusion Matrix - TF-IDF, Support Vector Machine', fontsize=18)
        plt.suptitle('Data: ' + file[17:file.find("_", 18)])

        plt.savefig('./Result_SupportVectorMachine/ConMat-TFIDF-SVM-test1-' + file[17:file.find("_", 18)] + '.png')

        print('Loading data...')
        splitlabels, datas, labels = read_data_with_labels(file_final_test)
        labels = [0 if x == 'de' else x for x in labels]
        labels = [1 if x == 'bar' else x for x in labels]
        labels = [2 if x == 'nds' else x for x in labels]
        print('Done!\n')

        data_test_tatoeba, labels_test_tatoeba = [], []

        # split into training data validation data and test data
        for splitlabel, data, label in zip(splitlabels, datas, labels):
            if splitlabel == 'test':
                data_test_tatoeba.append(data)
                labels_test_tatoeba.append(label)

        X_test_tf_tatoeba = tf_idf.transform(data_test_tatoeba)

        # load saved trained classifier
        print('Load Classifier...')
        model_loaded = load(classifier_name)
        print('Done!\n')

        # predict test data
        y_pred = model_loaded.predict(X_test_tf_tatoeba)
        print(classification_report(np.asarray(labels_test_tatoeba), y_pred))
        with open('./Result_SupportVectorMachine/classification_report_SVM_test2_' + file[17:file.find("_", 18)] + '.tsv', 'a', encoding='utf-16') as outfile:
            outfile.write(str(classification_report(np.asarray(labels_test_tatoeba), y_pred)))

        # save trained classifier
        print('Save Classifier...')
        clf = svm.SVC()
        X, y = datasets.load_iris(return_X_y=True)
        clf.fit(X, y)
        dump(clf, classifier_name)
        print('Done!\n')

        # Plot confusion matrix in a beautiful manner
        ax = plt.subplot()
        index = ['Hochdeutsch', 'Bairisch', 'Plattdeutsch']

        cnf_matrix = confusion_matrix(labels_validation, y_pred)
        df = pd.DataFrame(cnf_matrix, index=index, columns=index)
        sns.heatmap(df, annot=True, cmap='Blues', ax=ax)  # fmt='g'

        # labels, title and ticks
        ax.set_xlabel('Predicted', fontsize=15)
        ax.set_ylabel('True', fontsize=15)
        plt.title('Refined Confusion Matrix - TF-IDF, Support Vector Machine', fontsize=18)
        plt.suptitle('Data: ' + file[17:file.find("_", 18)])

        plt.savefig('./Result_SupportVectorMachine/ConMat-TFIDF-SVM-test2-' + file[17:file.find("_", 18)] + '.png')

