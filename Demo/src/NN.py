import auxilary
# import face_recognition
# import show_tests
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
    print (f'Inputs shape: {inputs.shape}')
    print (f'Outputs shape: {labels.shape}')
    return inputs, labels

def train():
    X, y = prepare_data()

    
if __name__ == "__main__":
    train()