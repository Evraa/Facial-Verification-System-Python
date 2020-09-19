from networkx.drawing.tests.test_pylab import plt
from sklearn.svm import SVC
from sympy.plotting.tests.test_plot import matplotlib

import auxilary
import numpy as np
from sklearn import svm, tree
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def svm_compare():
    '''
    TODO: implement svm
    '''
    path = '../csv_files/csv_differences_prof.csv'
    data = auxilary.read_csv(fileName=path)

    # READ THE INPUTS
    # we have 1520 inputs -> N
    # each one of them is 12 dimension -> D
    N = len(data)
    D = 12
    inputs = np.zeros([N, D])
    data_input = (data['inputs'])
    for i, input_list in enumerate(data_input):
        inputs[i] = auxilary.strings_to_lists(input_list)

    # READ THE LABELS
    data_labels = data['label'] == 0
    labels = np.ones([N, 1])
    labels[data_labels] = 0

    # clf = tree.DecisionTreeClassifier()
    clf = svm.SVC(gamma=0.001, C=100, probability=True)
    pca = PCA(n_components=2)
    X, y = inputs, np.ravel(labels)
    # shuffles the date to save 20% of data for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # TO GET EACH SAMPLE OUTPUT
    # for sample in X_test:
    #     y_pred = clf.predict_proba(sample.reshape(1, -1))
    #     print(y_pred)


    # TO DO ALL SAMPLES AT ONCE
    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    print("probabilities:\n", y_prob[:5])
    print("\npredictions: \n" , y_pred[:5])
    print("accuracy: \n",np.array(y_pred == y_test)[:5])
    print('\npercentage correct: ', 100 * np.sum(y_pred == y_test) / len(y_test))

    # Get support vectors themselves
    # support_vectors = clf.support_vectors_
    # X_train2 = pca.fit_transform(X_train)
    # clf.fit(X_train2, y_train)
    # plot_decision_regions(X_train2, y_train.astype(np.integer), clf=clf, legend=2)
    #
    # plt.xlabel(X_train[0], size=14)
    # plt.ylabel(X_train[1], size=14)
    # plt.title('SVM Decision Region Boundary', size=16)
    #
    # plt.show()

    show_vectors(X_train, y_train)

    return clf

def show_vectors(x, y):
    clf2 = svm.SVC(gamma=0.001, C=100, probability=True)
    pca = PCA(n_components=2)
    X_train2 = pca.fit_transform(x)
    clf2.fit(X_train2, y)
    plot_decision_regions(X_train2, y.astype(np.integer), clf=clf2, legend=2)


    plt.xlabel(x[0], size=14)
    plt.ylabel(x[1], size=14)
    plt.title('SVM Decision Region Boundary', size=16)

    plt.show()



