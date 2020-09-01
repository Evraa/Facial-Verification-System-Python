#GLOBAL IMPORTS
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import os
import keras
from keras.utils import np_utils 
from keras.datasets import mnist 
from keras.initializers import RandomNormal
from keras.initializers import he_normal
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation 
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout

import tensorflow
import matplotlib.pyplot as plt

#LOCAL IMPORTS
import auxilary


def prepare_data():
    print ("Load labels")
    
    data = auxilary.read_csv(fileName='../csv_files/embedded_2.csv')
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
    return inputs, y, names_encode

def split_data(X,y):
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X,
    #     y,
    #     test_size=0.2,
    #     shuffle=True,
    #     random_state=42,
    # )
    X_train = X
    y_train = y

    random_int = np.random.randint (0, X_train.shape[0]-1, size=(1500))
    X_test = X_train[random_int]
    y_test = y_train[random_int]
    print (f'Train data shape: {X_train.shape}')
    print (f'Test data shape: {X_test.shape}')
    print (f'Train output shape: {y_train.shape}')
    print (f'Test output shape: {y_test.shape}')
    
    return X_train, X_test, y_train, y_test

def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("../sequential_NN_model_2.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("../sequential_NN_model_2.h5")
    print("Saved model to disk")

def create_model_relu(input_dim,output_dim):

    model_relu = Sequential()
    model_relu.add(Dense(2048, activation='relu', input_shape=(input_dim,), kernel_initializer=tensorflow.keras.initializers.he_normal(seed=None)))
    model_relu.add(BatchNormalization())
    model_relu.add(Dropout(0.25))
    
    model_relu.add(Dense(1024, activation='relu', kernel_initializer=tensorflow.keras.initializers.he_normal(seed=None)))
    model_relu.add(BatchNormalization())
    model_relu.add(Dropout(0.25))
    
    model_relu.add(Dense(512, activation='relu', kernel_initializer=tensorflow.keras.initializers.he_normal(seed=None)) )
    model_relu.add(BatchNormalization())
    model_relu.add(Dropout(0.25))
    
    model_relu.add(Dense(256, activation='relu', kernel_initializer=tensorflow.keras.initializers.he_normal(seed=None)) )
    model_relu.add(BatchNormalization())
    model_relu.add(Dropout(0.25))
    

    model_relu.add(Dense(output_dim,activation='softmax'))
    print(model_relu.summary())
    model_relu.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model_relu
    
def load_model(second = False):
    if second:
        with open("../sequential_NN_model_2.json", "r") as json_file:
            json_loaded_model = json_file.read()
        model = model_from_json(json_loaded_model)

        model.load_weights('../sequential_NN_model_2.h5')
        return model
    else:
        with open("../sequential_NN_model.json", "r") as json_file:
            json_loaded_model = json_file.read()
        model = model_from_json(json_loaded_model)

        model.load_weights('../sequential_NN_model.h5')
        return model


def predict_input(embedding, second = False):
    embedding = np.reshape(embedding, (1,-1))
    inputs, y, le = prepare_data()
    model = load_model(second = second)
    pred = model.predict([[embedding]])
    # print (pred)
    ind = np.argsort(pred[0])
    print(ind[::-1][:5]) #FIRST five
    identical = []
    similars = []
    identical.append(le.inverse_transform([ind[::-1][0]])[0])
    for i in range (1,5):
        similars.append(le.inverse_transform([ind[::-1][i]])[0])
    # print("Prediction Probability: ",pred[0][ind[::-1][0]]*100,"%")
    print ("ID: ", identical)
    print ("Similar: ",similars)
    return identical, similars
    
def plot_acc (history):
    plt.figure(figsize=(16,5))
    plt.subplot(121)
    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')

    plt.show()
    return 

def train():
    if os.path.exists("../sequential_NN_model.h5"):
        print ("The model already exist, do you want to train it again?")
        input("Press Enter for re-training, press ctrl+c for exit")
    X, y, names_encode = prepare_data()
    X_train, X_val, y_train, y_val = split_data(X,y)
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model_relu = create_model_relu(input_dim,output_dim)
    history = model_relu.fit(X_train, y_train, batch_size=32, epochs=500,validation_data=(X_val,y_val))
    save_model(model_relu)
    plot_acc (history)
    return  model_relu


def predict_input_from_video(embedding, le, second= True):
    embedding = np.reshape(embedding, (1,-1))
    
    model = load_model(second = second)
    pred = model.predict([[embedding]])
    # print (pred)
    ind = np.argsort(pred[0])
    # print(ind[::-1][:5]) #FIRST five
    identical = []
    identical.append(le.inverse_transform([ind[::-1][0]])[0])
    percentage = pred[0][ind[::-1][0]]*100
    # print("Prediction Probability: ",pred[0][ind[::-1][0]]*100,"%")
    # print ("ID: ", identical)
    return identical[0], percentage
    