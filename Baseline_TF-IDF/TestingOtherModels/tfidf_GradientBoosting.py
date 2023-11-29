import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier  # https://scikit-learn.org/0.16/modules/ensemble.html
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
    data_test = data[30000:31000]
    labels_test = labels[30000:31000]

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

    # Classifier training Gradient Boosting Classifier
    print('Classifier training...')
    classifier = GradientBoostingClassifier()
    classifier.fit(X_train_tf, np.asarray(labels_train))
    print('Done!\n')

    # predict test data
    y_pred = classifier.predict(X_test_tf)
    print(classification_report(np.asarray(labels_test), y_pred))
