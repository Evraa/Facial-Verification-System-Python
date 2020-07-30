import auxilary
import numpy as np

def svm():
    '''
    TODO: implement svm
    '''
    path = '../csv_files/csv_differences.csv'
    data = auxilary.read_csv(fileName=path)

    #READ THE INPUTS
    #we have 1520 inputs -> N
    #each one of them is 12 dimension -> D
    N = len(data)
    D = 12
    inputs = np.zeros([N,D])
    data_input = (data['inputs'])
    for i,input_list in enumerate(data_input):
        inputs[i] = auxilary.strings_to_lists(input_list)
    
    #READ THE LABELS
    data_labels = data['label'] == 0
    labels = np.ones([N,1])
    labels[data_labels] = 0

