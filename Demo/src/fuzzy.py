#Global imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz

#Local imports
import auxilary 

#Global variables


def read_data():
    data = auxilary.read_csv(fileName='../csv_files/embedded.csv')
    N = data.shape[0] #5050
    D = data.shape[1] - 1 #22
    
    data_inputs = (data.iloc[:,:D])
    inputs = np.zeros([N,D])
    inputs = np.array(data_inputs)

    labels = np.zeros([N,1])
    labels = np.array(data.iloc[:,D])

    return inputs, labels

def cfuzzy(inputs, labels):
    inp_tr = np.transpose(inputs)
    unique_clusters = len(set(labels))
    print (inp_tr.shape)
    # colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

    # Define three cluster centers
    # centers = [[4, 2],
    #         [1, 7],
    #         [5, 6]]

    # Define three cluster sigmas in x and y, respectively
    # sigmas = [[0.8, 0.3],
    #         [0.3, 0.5],
    #         [1.1, 0.7]]

    # Generate test data
    # np.random.seed(42)  # Set seed for reproducibility
    # xpts = np.zeros(1)
    # ypts = np.zeros(1)
    # labels = np.zeros(1)
    # for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
    #     xpts = np.hstack((xpts, np.random.standard_normal(200) * xsigma + xmu))
    #     ypts = np.hstack((ypts, np.random.standard_normal(200) * ysigma + ymu))
        # labels = np.hstack((labels, np.ones(200) * i))

    # Visualize the test data
    # fig0, ax0 = plt.subplots()
    # for label in range(3):
    #     ax0.plot(xpts[labels == label], ypts[labels == label], '.',
    #             color=colors[label])
    # ax0.set_title('Test data: 200 points x3 clusters.')
    
    # alldata = np.vstack((xpts, ypts))
    # fpcs = []
    # fig1, axes1 = plt.subplots(2,2, figsize=(8, 8))

    print ("Starts...")
    fpcs = []
    xs = []
    for i in range (2,260, 50):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                inp_tr, i, 2, error=0.005, maxiter=1000, init=None)
        fpcs.append(fpc)
        xs.append(i)
        print (f'done with i = {i}, fpc: {fpc}')

    # fig2, ax2 = plt.subplots()
    # ax2.plot(xs, fpcs)
    # ax2.set_xlabel("Number of centers")
    # ax2.set_ylabel("Fuzzy partition coefficient")

    # for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    #     cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    #         alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

    #     fpcs.append(fpc)

    #     # Plot assigned clusters, for each data point in training set
    #     cluster_membership = np.argmax(u, axis=0)
    #     for j in range(ncenters):
    #         ax.plot(xpts[cluster_membership == j],
    #                 ypts[cluster_membership == j], '.', color=colors[j])

    #     # Mark the center of each fuzzy cluster
    #     for pt in cntr:
    #         ax.plot(pt[0], pt[1], 'rs')

    #     ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    #     ax.axis('off')

    # fig1.tight_layout()
    # plt.show()


if __name__ == "__main__":
    inputs, labels = read_data()
    
    cfuzzy(inputs, labels)