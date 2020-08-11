# from networkx.drawing.tests.test_pylab import plt
# from sympy.plotting.tests.test_plot import matplotlib

import auxilary
import numpy as np
from sklearn import svm, tree
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


def svm_compare():
    '''
    TODO: implement svm
    '''
    # path = '../csv_files/csv_differences.csv'
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
    X, y = inputs, np.ravel(labels)
    # shuffles the date to save 20% of data for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
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

    return clf



# svm_compare()
