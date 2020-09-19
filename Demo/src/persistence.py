import gudhi

import auxilary
import keras
import numpy as np
from auxilary import path_to_csv_key_points,read_csv
import matplotlib.pyplot as plt

import gudhi.representations as tda
from keras.layers import Embedding
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.model_selection import GridSearchCV, train_test_split

data = read_csv(fileName=path_to_csv_key_points)
labels = data["image_set"].dropna().to_list()
features = data[data.columns[3:15]].dropna()


features = auxilary.get_enum_coor(features)


# pers_diag = [(1, elt) for elt in features]
# # Use subplots to display diagram and density side by side
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
# gudhi.plot_persistence_diagram(persistence=pers_diag,
#     axes=axes[0])
# gudhi.plot_persistence_density(persistence=pers_diag,
#     dimension=1, legend=True, axes=axes[1])
# plt.show()

#
#
# rips = gudhi.RipsComplex(points=features).create_simplex_tree()
# dgm = rips.persistence()
#
# gudhi.plot_persistence_diagram(persistence=dgm, alpha=.6, band=0.1, max_intervals=1000, max_plots=1000, inf_delta=0.05, legend=False)
# plt.show()
#
# print((dgm))


rips_complex = gudhi.RipsComplex(points=features,
                                 max_edge_length=2000)

simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
result_str = 'Rips complex is of dimension ' + repr(simplex_tree.dimension()) + ' - ' + \
    repr(simplex_tree.num_simplices()) + ' simplices - ' + \
    repr(simplex_tree.num_vertices()) + ' vertices.'
print(result_str)
fmt = '%s -> %.2f'
for filtered_value in simplex_tree.get_filtration():
    print(fmt % tuple(filtered_value))

pipe = Pipeline([("TDA", tda.PersistenceImage()),
                 ("Estimator", SVC())])

param = [
    # {"TDA": [tda.SlicedWassersteinKernel()],
    #       "TDA__bandwidth": [0.1, 1.0],
    #       "TDA__num_directions": [20],
    #       "Estimator": [SVC(kernel="precomputed")]},

         {"TDA": [tda.PersistenceWeightedGaussianKernel()],
          "TDA__bandwidth": [0.1, 0.01],
          "TDA__weight": [lambda x: np.arctan(x[1] - x[0])],
          "Estimator": [SVC(kernel="precomputed")]},

         {"TDA": [tda.PersistenceImage()],
          "TDA__resolution": [[5, 5], [6, 6]],
          "TDA__bandwidth": [0.01, 0.1, 1.0, 10.0],
          "Estimator": [SVC()]},

         {"TDA": [tda.Landscape()],
          "TDA__resolution": [100],
          "Estimator": [RF()]},

         {"TDA": [tda.BottleneckDistance()],
          "TDA__epsilon": [0.1],
          "Estimator": [kNN(metric="precomputed")]}
         ]
model = GridSearchCV(pipe, param, cv=3)
inputs = keras.Input(shape=(12,))
embedding = Embedding(161, 15)
X = embedding(inputs)
# for i in range(len(labels)):
#     labels[i] = np.full(shape=12, fill_value=labels[i], dtype=np.float)
labels = np.array(labels)
features = np.array(features)


# print(features.shape, labels.shape)

model = model.fit(features, labels)