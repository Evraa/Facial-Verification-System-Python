# from auxilary import path_to_all_dataset,os,path_to_yalefaces
# from facial_landmarks import draw_landmarks, draw_parts,store_key_points
# import facial_landmarks
# import auxilary
# import os
# import face_recognition
# import SVM
# import show_tests
# from get_lengths import *
# from identify_faces import *
# from calc_weights import calc_weights
# from delaunay import get_delaunay_points,store_shape_tri
import face_recognition
from joblib import dump, load
import auxilary
import pandas as pd
import numpy as np
import copy
import show_tests
import os
'''
    Jobs allowed:
        + show an image with facial landmarks on it (dominance option)
        + show an image with drawings on it
        + store key points
        + store lengths...done and wont need it anymore, but it's still there!
        + calc weights
        
'''


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

def trim_outputs (labels, face_name, identicals, similars):
    N = labels.shape[0]
    others_names = []
    others_paths = []
    path = '../dataset/lfw/'
    #for others
    for i in range (9):
        rand_int = np.random.random_integers(0,N-1)
        label = labels[rand_int]
        dir_path = path + label + '/'
        files_count = (os.listdir(dir_path))
        while label == face_name or len(files_count) < 1 or \
            label in identicals or label in similars:
            
            rand_int = np.random.random_integers(0,N-1)
            label = labels[rand_int]
            dir_path = path + label + '/'
            files_count = (os.listdir(dir_path))

        others_names.append(label)
        others_paths.append(dir_path + files_count[0])

    #SIMILARS
    similars_2 = []
    for i in range(len(similars)):
        if len(similars_2) == 9:
            break
        rand_int = np.random.random_integers(0,len(similars))
        sim = similars[i]
        if sim not in similars_2:
            similars_2.append(sim)

    #then, it's good
    sim_paths = []
    sim_names = similars_2
    for sim in similars_2:
        dir_path = path + sim + '/'
        files_count = (os.listdir(dir_path))
        sim_paths.append(dir_path + files_count[0])

    
    #IDENTICALS
    idc_paths, idc_names = [], []
    #make sure the main file occurs
    dir_path = path + face_name + '/'
    files_count = os.listdir(dir_path)
    for i, file_count in enumerate(files_count):
        if i >= 9:
            break
        idc_names.append(face_name)
        idc_paths.append(dir_path + file_count)
        if face_name in identicals:
            identicals.remove(face_name)

    others_count = 9 - len(idc_names)

    for i in range(len(identicals)):
        if others_count == 0:
            break
        rand_int = np.random.random_integers(0,len(identicals))
        idc = identicals[i]
        if idc not in idc_names:
            idc_names.append("NOT MATCHING")
            dir_path = path + idc + '/'
            files_count = (os.listdir(dir_path))
            idc_paths.append(dir_path + files_count[0])
            others_count -= 1

    print ('\n')
    print (idc_names)
    print (idc_paths)
    print ("\n\n\n")
    print (sim_names)
    print (sim_paths)
    print ("\n\n\n")
    print (others_names)
    print (others_paths)
    print ("\n\n\n")
    return idc_paths, idc_names, sim_paths, sim_names, others_paths, others_names

if __name__ == "__main__":
    print("hello :D")
    # SVM.svm_compare(path="../csv_files/embedded.csv")
    # image_path = '../dataset/Mag.jpg'
    # # facial_landmarks.draw_landmarks(image_path)
    # # facial_landmarks.draw_parts(image_path)
    
    # print ("Loading the detector and predictor...\n")
    # predictor , detector = facial_landmarks.load_pred_detec(auxilary.path_to_shape_predictor)
    # print ("Training the Classifier...\n")
    # clf = SVM.svm_compare()
    # print ("Testing random image...\n")
    # show_tests.show_tests(auxilary.path_to_yalefaces,clf,detector,predictor)

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
    #KR: 7355
    #LBJ: 7820
    #KG: 7443
    embeddings,face_name, human_file_path = face_recognition.face_recognition(dataset_path = "../dataset/lfw/*/*", preview=True, image_num = 7355)
    identicals, similars = results(embeddings, inputs, labels)



    idc_paths, idc_names, sim_paths, sim_names, others_paths, others_names = \
        trim_outputs (labels, face_name, identicals, similars)

    show_tests.buttons(idc_paths, idc_names, sim_paths,sim_names, others_paths, others_names,
                human_file_path, face_name,"MATCHING", "SIMILARS", "OTHERS")