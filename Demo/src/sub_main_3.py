'''
    Finds the threshold for similar/identical/non matching images.
'''



import pandas as pd
import auxilary
import numpy as np
import os
import matplotlib.pyplot as plt


def euclidean(input_1, input_2):
    diff = input_1 - input_2
    power = np.power(diff, 2)
    return np.sqrt(np.sum(power))
    
print ("Load Data")
data = auxilary.read_csv(fileName='../csv_files/embedded.csv')
N = len(data) #13142
D = 128

#READING THE DATA

#MOST IMAGES 530 at George_W_Bush
#Largest diff = 1.57
# data_GB = data.iloc[:,D]

# data_GB = data[data["output"] == "George_W_Bush"]

data_inputs = (data.iloc[:,:D])
inputs = np.zeros([N,D])
inputs = np.array(data_inputs)

labels = np.zeros([N,1])
labels = np.array(data.iloc[:,D])


Eucs = []
for i in range (140185):
    if i%10000 == 0:
        print (f'iteraiton: {i}')
    #pick two random rows that are not equal
    rand_int_1 = np.random.random_integers(0,N-1)
    rand_int_2 = np.random.random_integers(0,N-1)
    while rand_int_1 == rand_int_2:
        rand_int_1 = np.random.random_integers(0,N-1)
        rand_int_2 = np.random.random_integers(0,N-1)
    try:

        input_1 = data_inputs.iloc[rand_int_1,:]
        input_2 = data_inputs.iloc[rand_int_2,:]
        Eucs.append(euclidean(input_1, input_2))
    except:
        print (rand_int_1)
Eucs_2 = []
inputs_GB = data[data["output"] == "George_W_Bush"]
data_inputs = (inputs_GB.iloc[:,:D])
inputs_GB = np.zeros([N,D])
inputs_GB = np.array(data_inputs)

for i, input_1 in enumerate(inputs_GB):
    print (f'iteraiton: {i}')
    for j, input_2 in enumerate(inputs_GB):
        if i >= j:
            continue
        Eucs_2.append(euclidean(input_1, input_2))



Eucs = sorted(Eucs,reverse=True)
lenh =  (len(Eucs))
Eucs_2 = sorted(Eucs_2,reverse=True)
lenh_2 =  (len(Eucs_2))
xs = np.linspace(0,lenh,lenh)
xs_2 = np.linspace(0,lenh_2,lenh_2)

# plt.plot(xs,Eucs)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(xs, Eucs)
ax1.set_title('Different images Euc. differences')
ax2.plot(xs_2, Eucs_2)
ax2.set_title('Similar images Euc. differences')



plt.show()



print (Eucs[:20])
print (Eucs[lenh-20:lenh])
print (np.max(Eucs))
print (np.min(Eucs))
# direc = '../dataset/lfw_affine/'
# most = 0
# most_one = None
# for label in labels:
#     path = direc + label + '/'
#     files = len(os.listdir(path))
#     if files > most:
#         most = files
#         most_one = label
# print (f'count: {most} person {most_one}')