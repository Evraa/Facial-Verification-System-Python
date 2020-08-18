# gaussian mixture clustering
import numpy as np
from numpy import where, unique
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot
import auxilary
from sklearn.metrics import accuracy_score


path = '../csv_files/csv_key_points.csv'
data = auxilary.read_csv(fileName=path)
alldata = data.iloc[:, 2:].to_numpy()
labels = data["image_set"].tolist()
groups = []
for sets in alldata:
    coords = []
    for i in sets:
        test_tup = []
        for j in i.split():
            test_tup.append(j.strip('[],'))
        coords.append(float('.'.join(str(ele) for ele in test_tup)))
    groups.append(coords)
X = np.asarray(groups)

# define the model
model = GaussianMixture(n_components=15)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster

acc=accuracy_score(labels, model)
print("Accuracy score is", acc)

for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.title("Gaussian Mixture")
pyplot.show()



# synthetic classification dataset
from numpy import where
from matplotlib import pyplot

# define dataset
# create scatter plot for samples from each class
for class_value in range(15):
    # get row indexes for samples with this class
    row_ix = [labels[row_ix] == class_value for row_ix in labels]
    # row_ix = where(labels == class_value)
    # create scatter of these samples
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
