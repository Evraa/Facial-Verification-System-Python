# from networkx.drawing.tests.test_pylab import plt
# from sympy.plotting.tests.test_plot import matplotlib
# from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import auxilary
import numpy as np
from sklearn import svm, datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


def multiclass_classification(path):
    dataframe = auxilary.read_csv(path)

    # Store variables as target y and the first two features as X (sepal length and sepal width of the iris flowers)
    X = dataframe.iloc[:, 0:22].to_numpy()
    y = dataframe['output'].to_list()


    decision_tree(X, y)


def svm_multi():

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

    linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
    rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(X_train, y_train)
    poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train, y_train)
    sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(X_train, y_train)

    # stepsize in the mesh, it alters the accuracy of the plotprint
    # to better understand it, just play with the value, change it and print it
    h = .01

    # create the mesh
    def create_mesh(num_meshes):
        list_mesh = []
        for n in range(num_meshes):
            list_mesh.append([X[:, n].min() - 1, X[:, n].max() + 1])
        return list_mesh

    def meshgrid_list(list_of_ranges):
        list_of_meshes = []
        for pair in list_of_ranges:
            list_of_meshes.append(np.arange(pair[0], pair[1], h))
        return list_of_meshes

    meshes = meshgrid_list(create_mesh(22))

    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22 = np.meshgrid(
        *meshes)
    # create the title that will be shown on the plot
    titles = ['Linear kernel', 'RBF kernel', 'Polynomial kernel', 'Sigmoid kernel']

    print(f1, f2)

    for i, clf in enumerate((linear, rbf, poly, sig)):
        # defines how many plots: 2 rows, 2columns=> leading to 4 plots
        plt.subplot(2, 2, i + 1)  # i+1 is the index
        # space between plots
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.PuBuGn, alpha=0.7)
        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.PuBuGn, edgecolors='grey')
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])
        plt.show()


def decision_tree(X, y):

    print(X, "\n\n\n", y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

    # X_train =

    # training a DescisionTreeClassifier
    from sklearn.tree import DecisionTreeClassifier
    dtree_model = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
    dtree_predictions = dtree_model.predict(X_test)

    # creating a confusion matrix
    cm = confusion_matrix(y_test, dtree_predictions)
    print("PREDICTIONS\n\n",dtree_predictions)
    print(cm.shape)

def svm_compare():
    '''
    TODO: implement svm
    '''
    # path = '../csv_files/csv_differences.csv'
    path = '../csv_files/svm_set2.csv'
    data = auxilary.read_csv(fileName=path)

    # READ THE INPUTS
    # we have 1520 inputs -> N
    # each one of them is 12 dimension -> D
    N = len(data)
    D = 22
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

# svm_compare()
# multiclass_classification('../csv_files/embedded_2.csv')