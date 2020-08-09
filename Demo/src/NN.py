import auxilary
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os


def prepare_data():
    print ("Load labels")
    data = auxilary.read_csv(fileName='../csv_files/embedded.csv')
    N = data.shape[0] #5050
    D = data.shape[1] - 1 #128
    data_inputs = (data.iloc[:,:D])
    inputs = np.zeros([N,D])
    inputs = np.array(data_inputs)

    labels = np.zeros([N,1])
    labels = np.array(data.iloc[:,D])

    names_encode = LabelEncoder().fit(labels)
    Y = names_encode.transform(labels) # in range of [0:252]
    unique_count = len(set(labels))
    #Transofrm the Y into shape of [5050, 253] with all equals zeros, except for the correct label
    y = np.zeros([N,unique_count])
    for i, row in enumerate(y):
        row[Y[i]] = 1

    print (f'Inputs shape: {inputs.shape}')
    print (f'Outputs shape: {y.shape}')
    return inputs, y

def train():
    X, y = prepare_data()

    

if __name__ == "__main__":
    train()