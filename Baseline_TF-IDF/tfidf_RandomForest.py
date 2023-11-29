import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from read_data import read_data, read_data_with_labels
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix, classification_report


if __name__ == '__main__':

    files = ['../Data_Training/Dialektversum_de+nds+bar_mixed_preprocessed.tsv',
             '../DataTraining/MrDialect_de+nds+bar_mixed_preprocessed.tsv',
             '../DataTraining/Wikipedia_de+nds+bar_mixed.tsv']

    for file in files:

        print('Loading data...')
        splitlabels, datas, labels = read_data_with_labels(file)
        labels = [0 if x == 'de' else x for x in labels]
        labels = [1 if x == 'bar' else x for x in labels]
        labels = [2 if x == 'nds' else x for x in labels]
        print('Done!\n')

        # split into training data validation data and test data
        data_train, labels_train, data_validation, labels_validation, data_test, labels_test = []
        for splitlabel, data, label in zip(splitlabels, datas, labels):
            if splitlabel == 'train':
                data_train.append(data)
                labels_train.append(label)
            elif splitlabel == 'validation':
                data_validation.append(data)
                labels_validation.append(label)
            elif splitlabel == 'test':
                data_test.append(data)
                labels_test.append(label)

        # vectorize the data
        print('Vectorize data...')
        tf_idf = TfidfVectorizer()
        X_train_tf = tf_idf.fit_transform(data_train)
        X_test_tf = tf_idf.transform(data_test)
        print('Done!\n')

        # dimensionality reduction using PCA
        print('Dimensionality reduction...')
        pca = TruncatedSVD(n_components=128)
        X_train_tf_pca = pca.fit_transform(X_train_tf, None)
        print('Done!\n')

        # Classifier training RandomForest
        print('Classifier training...')
        classifier = RandomForestClassifier(verbose=1)
        classifier.fit(X_train_tf, np.asarray(labels_train))
        print('Done!\n')

        # predict test data
        y_pred = classifier.predict(X_test_tf)
        print(classification_report(np.asarray(labels_test), y_pred))

        # Plot confusion matrix in a beautiful manner
        ax = plt.subplot()
        index = ['Hochdeutsch', 'Bairisch', 'Plattdeutsch']

        cnf_matrix = confusion_matrix(labels_test, y_pred)
        df = pd.DataFrame(cnf_matrix, index=index, columns=index)
        sns.heatmap(df, annot=True, cmap='Blues', ax=ax) # fmt='g'

        # labels, title and ticks
        ax.set_xlabel('Predicted', fontsize=15)
        ax.set_ylabel('True', fontsize=15)
        plt.title('Refined Confusion Matrix - TF-IDF, Random Forest', fontsize=18)
        plt.suptitle('Data: ' + file[17:file.find("_", 18)])

        plt.savefig('./Result_SupportVectorMachine/ConMat-TFIDF-RF-' + file[17:file.find("_", 18)] + '.png')


