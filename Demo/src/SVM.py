import auxilary
import numpy as np
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


def svm_compare():
    '''
    TODO: implement svm
    '''
    path = '../csv_files/csv_differences.csv'
    data = auxilary.read_csv(fileName=path)

    #READ THE INPUTS
    #we have 1520 inputs -> N
    #each one of them is 12 dimension -> D
    N = len(data)
    D = 12
    inputs = np.zeros([N,D])
    data_input = (data['inputs'])
    for i,input_list in enumerate(data_input):
        inputs[i] = auxilary.strings_to_lists(input_list)
    
    #READ THE LABELS
    data_labels = data['label'] == 0
    labels = np.ones([N,1])
    labels[data_labels] = 0

    clf = svm.SVC(gamma=0.001, C=100)
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

    y_pred = clf.predict(X_test)
    print("predictions: \n" , y_pred)
    print("\naccuracy: \n",np.array(y_pred == y_test)[:25])
    print('\npercentage correct: ', 100 * np.sum(y_pred == y_test) / len(y_test))


svm_compare()
