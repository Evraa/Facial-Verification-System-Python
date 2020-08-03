import face_recognition
from joblib import dump, load
import auxilary
import pandas as pd
import numpy as np
import copy

def euclidean(input_1, input_2):
    diff = input_1 - input_2
    power = np.power(diff, 2)
    return np.sqrt(np.sum(power))

def results(embeddings, inputs, labels):
    identicals = []
    similars = []
    for i, input_elm in enumerate(inputs):
        error = euclidean(embeddings, input_elm)
        if error >= 0.8:
            continue
        if error < 0.8 and error >= 0.75:
            similars.append(labels[i])
        if error < 0.75:
            identicals.append(labels[i])
        
    return identicals, similars


if __name__ == "__main__":
    print ("hello :D")

    print ("Load labels")
    data = auxilary.read_csv(fileName='../csv_files/embedded.csv')
    N = len(data) #13142
    D = 128
    
    data_inputs = (data.iloc[:,:D])
    inputs = np.zeros([N,D])
    inputs = np.array(data_inputs)

    labels = np.zeros([N,1])
    labels = np.array(data.iloc[:,D])

    embeddings,face_name = face_recognition.face_recognition(dataset_path = "../dataset/lfw/*/*", preview=True)
    identicals, similars = results(embeddings, inputs, labels)

    print (f'face is: {face_name}')

    print (identicals)

    print (similars)
    # for label in labels:
    #     if label not in labels_unique:
    #         labels_unique.append(label)

    # embeddings,face_name = face_recognition.face_recognition(dataset_path = "../dataset/lfw/*/*", preview=True)
    # embeddings = embeddings.reshape(1,-1)
    # print ("Loading classifier.....")
    # clf = load(auxilary.path_to_clf)
    # print ("Predicting the face")
    # # print (clf.get_params())
    # # input("E")
    # results = clf.predict(embeddings)
    # proba = clf.predict_proba(embeddings)
    # prob = np.copy(proba[0])
    # if np.max(prob) < 0.005:
    #     print ("Falsly predicted!!!!")
    # print (f'face belongs to: {face_name}')
    # print (f'Predicted: {results}')
    # print (prob.shape)

    # # prob = np.sort(prob)
    # # prob = np.flip(prob)
    # # print (prob[:10])
    # maxy  = np.max(prob)
    # print (maxy)
    # indx = np.where(prob==maxy)
    # print (indx[0][0])
    # print (labels_unique[indx[0][0]-1])
    
    # print (labels_unique[indx[0][0]])

    # print (labels_unique[indx[0][0]+1])
    
    # for i in range (10):
    #     element = prob[i]
    #     indx = (np.where(proba[0] == element))

    #     print (dataset_names[indx[0]])
    

    # df = pd.DataFrame(proba)
    # df.to_csv("../csv_files/test.csv",index=False)

# listy =  [1,3,4,5,68,1,34,3,67,8,6,6,5,42,1,4,5,6,7,9,10]
# indx = [1,3]
# print (listy.index(indx))