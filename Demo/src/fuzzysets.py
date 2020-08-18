from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import auxilary

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

path = '../csv_files/csv_key_points.csv'
data = auxilary.read_csv(fileName=path)
alldata = data.iloc[:, 2:].to_numpy()
labels = data["image_set"]
groups = []
for sets in alldata:
    coords = []
    for i in sets:
        test_tup = []
        for j in i.split():
            test_tup.append(j.strip('[],'))
        coords.append(float('.'.join(str(ele) for ele in test_tup)))
    groups.append(coords)
alldata = np.asarray(groups)

# # Set up the loop and plot
# fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
# # alldata = np.vstack((xpts, ypts))
# fpcs = []
#
# for ncenters, ax in enumerate(axes1.reshape(-1), 2):
#     cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
#         alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)
#
#     # Store fpc values for later
#     fpcs.append(fpc)
#
#     # Plot assigned clusters, for each data point in training set
#     cluster_membership = np.argmax(u, axis=0)
#     for j in range(ncenters):
#         # ax.plot(xpts[cluster_membership == j],
#         #         ypts[cluster_membership == j], '.', color=colors[j])
#         ax.plot([0, 1, 2, 3 ,4,5,6,7,8,9,10],
#                 [0, 1, 2, 3 ,4,5,6,7,8,9,10], '.', color=colors[j])
#
#     # Mark the center of each fuzzy cluster
#     for pt in cntr:
#         ax.plot(pt[0], pt[1], 'rs')
#
#     ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
#     ax.axis('off')
#
# fig1.tight_layout()


cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
    alldata, 15, 2, error=0.005, maxiter=1000)

# Show 3-cluster model
fig2, ax2 = plt.subplots()
ax2.set_title('Trained model')
for j in range(15):
    ax2.plot(alldata[0, u_orig.argmax(axis=0) == j],
             alldata[1, u_orig.argmax(axis=0) == j], 'o',
             label='series ' + str(j))

ax2.legend()

# Generate uniformly sampled data spread across the range [0, 10] in x and y
# newdata = np.random.uniform(0, 1,(161, 13)) * 10
# print(newdata.shape)
#
# # Predict new cluster membership with `cmeans_predict` as well as
# # `cntr` from the 3-cluster model
# u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
#     newdata.T, cntr, 2, error=0.005, maxiter=1000)
#
# # Plot the classified uniform data. Note for visualization the maximum
# # membership value has been taken at each point (i.e. these are hardened,
# # not fuzzy results visualized) but the full fuzzy result is the output
# # from cmeans_predict.
# cluster_membership = np.argmax(u, axis=0)  # Hardening for visualization
#
# fig3, ax3 = plt.subplots()
# ax3.set_title('Random points classifed according to known centers')
# for j in range(15):
#     ax3.plot(newdata[cluster_membership == j, 0],
#              newdata[cluster_membership == j, 1], 'o',
#              label='series ' + str(j))
# ax3.legend()


plt.show()