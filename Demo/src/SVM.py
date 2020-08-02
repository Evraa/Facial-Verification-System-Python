import auxilary
import numpy as np
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load
import random


def svm_compare(path = '../csv_files/csv_differences.csv'):
    '''
    TODO: implement svm
    '''
    
    data = auxilary.read_csv(fileName=path)
    N = len(data) #13142
    D = 128

    #READING THE DATA
    data_inputs = (data.iloc[:,:D])
    inputs = np.zeros([N,D])
    inputs = np.array(data_inputs)
    
    labels = np.zeros([D,1])
    labels = np.array(data.iloc[:,D])
    #Creating classifier
    best_clf = None
    best_acr = -1
    scale_gammas = np.linspace(0,-10,10)

    # for scale_gamma in scale_gammas:
    regz = 100
    gamma = 0.1
    clf = svm.SVC(kernel="rbf", gamma=gamma, C=regz)
    
    # val_portion = 1000
    # X, y = inputs[:val_portion,:],labels[:val_portion]
    shuffled_list = list(range(N))
    random.shuffle(shuffled_list)
    random_list = list(np.random.randint(0, N, size=1000))
    
    X_train,y_train = inputs[shuffled_list,:], labels[shuffled_list]
    X_test, y_test  = inputs[random_list,:] ,labels[random_list]

    
    # # shuffles the date to save 20% of data for testing
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X,
    #     y,
    #     test_size=0.1,
    #     shuffle=True,
    #     random_state=42,
    # )

    print (f'Training data shape: {X_train.shape}')
    # print (f'Test data shape: {X_test.shape}')
    print ("Start classifying, This may take a while.....")
    clf.fit(X_train, y_train)
    
    # TO GET EACH SAMPLE OUTPUT
    # for sample in X_test:
    #     y_pred = clf.predict_proba(sample.reshape(1, -1))
    #     print(y_pred)
    
    # TO DO ALL SAMPLES AT ONCE
    # y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    # print(y_prob[:5])
    # print("predictions: \n" , y_pred[:5])
    # print("\naccuracy: \n",np.array(y_pred == y_test)[:5])
    # accuracy = np.sum(y_pred == y_test) / len(y_test)
    # print (f"with gamma = {gamma}")
    print('percentage correct: ', 100 * np.sum(y_pred == y_test) / len(y_test))
    # print ("\n")
    # if accuracy > best_acr:
    #     best_clf = clf
    #     best_acr = accuracy

    dump(clf, 'best_SVM_clf_0.joblib')

    # return clf

# svm_compare()

