'''
    Simply taking the row of features, it returns the most relevant result.
'''

#Global imports
import numpy as np

#Local imports


#Global Variables
weights = [ 0 ,2.52908643  ,0.86726481  ,0.94175242 ,10.69470033  ,2.63917037
  ,2.76546006  ,1.47907017  ,1.67832444  ,3.83664557  ,1.7575141   ,1.61621531
  ,0.77266737  ,6.73037417  ,3.88943935  ,3.98608721  ,7.41608733  ,2.61684973
  ,2.64867014 ,18.84702478 ,16.84196087  ,5.44563503]


def euc_predict(embeddings, inputs, labels):
    min_dist = 100
    correct = None
    for emb, label in zip(inputs, labels):
        poww = np.power((embeddings - emb),2)
        weighted = np.multiply(poww, weights)
        dist = (np.sum(weighted))
        if dist != 0 and dist < min_dist:
            correct = label
            min_dist = dist

    return correct, 1- dist