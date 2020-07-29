from sklearn import svm

X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)

print("predict: ", clf.predict([[-1, 3]]))

# get support vectors
print("vectors: ", clf.support_vectors_)

# get indices of support vectors
print("vector indices: ", clf.support_)

# get number of support vectors for each class
print("num vectors: ", clf.n_support_)
