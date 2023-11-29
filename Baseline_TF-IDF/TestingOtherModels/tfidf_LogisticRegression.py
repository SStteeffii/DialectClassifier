import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from read_data import read_data, read_data_with_labels
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix, classification_report


if __name__ == '__main__':

    print('Loading data...')
    data, labels = read_data_with_labels('data_mixed_labeled_preprocessed.tsv')
    labels = [0 if x == 'de' else x for x in labels]
    labels = [1 if x == 'bar' else x for x in labels]
    labels = [2 if x == 'nds' else x for x in labels]
    print('Done!\n')

    # split into training data and test data
    data_train = data[:30000]
    labels_train = labels[:30000]
    data_valid = data[30000:31000]
    labels_valid = labels[30000:31000]
    data_test = data[31000:32000]
    labels_test = labels[31000:32000]

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

    # Classifier Logistic Regression
    print('Classifier training...')
    classifier = LogisticRegression(C=5e1, solver='lbfgs', multi_class='multinomial', random_state=17, n_jobs=4)
    print('Done!\n')

    classifier.fit(X_train_tf, np.asarray(labels_train))

    # predict test data
    y_pred = classifier.predict(X_test_tf)
    print(classification_report(np.asarray(labels_test), y_pred))

    cnf_matrix = metrics.confusion_matrix(np.asarray(labels_test), y_pred)
    print(cnf_matrix)
