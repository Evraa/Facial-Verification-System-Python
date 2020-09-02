'''
    This script tests the main_data file with:
        1- NN
        2- svm
        3- Fuzzy
        4- Euclidean
    
    For both the manual and imageNet models
'''
#GLOBAL IMPORTS
from glob import glob
import sys
sys.path.append('../src/')
import numpy as np
###################################for test_nn uncomment these#############################
# from create_model import create_model
# import openface
# import cv2
import pandas as pd

#LOCAL IMPORTS
###################################for test_nn uncomment these#############################
# import NN
# import face_recognition
# import facial_landmarks
import euclidean
import auxilary

#GLOBAL VARs
###################################for test_nn uncomment these#############################
# model_affine = create_model()
# model_affine.load_weights('../open_face.h5')
# pred, detc = facial_landmarks.load_pred_detec()


def get_the_name (s):
    s = s.split('/')
    s = s[-1]
    s = s.split('\\')
    return s[1]

def test_nn(data_path, second = False):
    full_data = np.array(glob(data_path))
    true = 0
    count = 0
    percentages = []
    labels = []
    model_image = NN.load_model(second = second) 
    for img in full_data:
        #Read the image
        state, shape, rect, image = facial_landmarks.get_shape(img, pred, detc)
        if not state:
            continue
        if not second:
            embeddings = face_recognition.get_embeddings(image, rect, model_affine)
            name, percent = NN.predict_input_with_percentage(embeddings, second = second, model=model_image)
        else:
            _, embeddings = facial_landmarks.get_ratios(shape, image)
            name, percent = NN.predict_input_with_percentage(embeddings, second = second, model=model_image)
        count += 1
        labels.append(get_the_name(img))
        if name == get_the_name(img):
            true += 1
            percentages.append(percent)
        else:
            percentages.append(-percent)
        if count % 100 == 0:
            print (f'Working, iteration {count}')
            print (f'Accuracy = {true/count}')

    df = pd.DataFrame(labels)
    df["NN_Manual_perc"] = percentages
    df.to_csv("../csv_files/PERC_1.csv",index=False)
    percentages = np.array(percentages)
    print (f"Accuracy: {true/count}")
    print (f'detected with prob = {np.average(percentages)}')

def test_euc():
    data = auxilary.read_csv(fileName='../csv_files/embedded_2.csv')
    N = data.shape[0] #5050
    D = data.shape[1] - 1 #22
    
    data_inputs = (data.iloc[:,:D])
    inputs = np.zeros([N,D])
    inputs = np.array(data_inputs)

    labels = np.zeros([N,1])
    labels = np.array(data.iloc[:,D])
    
    true = 0
    count = 0
    percentages = []

    for emb, label in zip(inputs, labels):
        name, perc = euclidean.euc_predict(emb, inputs, labels)

        if name == label:
            true += 1
            percentages.append(perc)
        else:
            percentages.append(-perc)
        count += 1
        if count % 100 == 0:
            print (f'Working, iteration {count}')
            print (f'Accuracy = {true/count}')

    percentages = np.array(percentages)
    print (f"Accuracy: {true/count}")
    print (f'detected with prob = {np.average(percentages)}')

if __name__ == "__main__":
    main_path = '../dataset/main_data/*/*'
    # test_nn(data_path = main_path, second=False)
    test_euc()