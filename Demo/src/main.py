import facial_landmarks
# import auxilary
import face_recognition
import SVM
import show_tests
# from get_lengths import *
# from identify_faces import *
# from calc_weights import calc_weights
# from delaunay import get_delaunay_points,store_shape_tri
# import face_recognition
# from joblib import dump, load
import auxilary
# import pandas as pd
import numpy as np
from termcolor import colored
from glob import glob
from create_model import create_model
import euc
import NN


def take_action():
    print (colored("\t\t\tWelcome to our humble applicaion", 'yellow'))
    print (colored("\tBefore we start, please make sure that your dataset is placed at Demo/dataset/*",'red'))
    print (colored("\tand the two models are placed at Demo/src/",'red'))
    
    print (colored("\tTo Transform your data to affine dataset, \t\tpress 1",'cyan'))
    print (colored("\tTo Create Embeddings for the data you just transformed, press 2",'cyan'))
    print (colored("\tTo Test classification using Euclidean equation, \tpress 3",'cyan'))
    print (colored("\tTo Train the NN model for classification, \t\tpress 4",'cyan'))
    print (colored("\tTo Test classification using Neural Networks, \t\tpress 5",'cyan'))

    print (colored("\tTo Exit \t\t\t\t\t\tpress 0",'cyan'))
    exit = False
    while not exit:
        try:
            action = int(input(colored("\tYour action: ", "green")))
            exit = True
        except:
            print (colored('\tPlease enter a number as specified','red'))
   
    return action

if __name__ == "__main__":
    # Show Welcome message with available options
    # Available options:
        # 1- Transofrm images at the dataset to be at lfw/affine/
        # 2- Create Embeddings from these images at csv_files/embeddings.csv
        # 3- Apply classification using Euclidean metric to test one image
        # 4- Apply classification using Neural Networks

    print (colored('\t\tLoading models once, to make the rest of the operations faster','yellow'))
    pred, detc = facial_landmarks.load_pred_detec()
    while True:
        action = take_action()

        if action == 1:
            #Affine transformation
            print (colored("\t\t\tSTARTING",'green'))
            dataset_path = '../dataset/main_data/*/*'
            human_files = np.array(glob(dataset_path))
            try:
                face_recognition.affine_transformation(human_files,pred,detc)
                print (colored('\t\t\tDONE','green'))
            except:
                print (colored("\t\t\tERROR",'red'))
            
        elif action == 2:
            dataset_path = '../dataset/lfw_affine/*/*'
            human_files = np.array(glob(dataset_path))
            model = create_model()
            model.load_weights('../open_face.h5')
            print (colored("\t\t\tSTARTING",'green'))            
            try:
                face_recognition.store_embeddings(human_files,model)
                print (colored('\t\t\tDONE','green'))
            except:
                print (colored("\t\t\tERROR",'red'))
        
        elif action == 3:
            print (colored("\t\t\tSTARTING",'green'))            
            try:
                # image_num = 7820
                euc.Euc_result_preview()
                print (colored('\t\t\tDONE','green'))
            except:
                print (colored("\t\t\tERROR",'red'))
        
        elif action == 4:
            print (colored("\t\t\tSTARTING",'green'))            
            try:
                NN.train()
                print (colored('\t\t\tDONE','green'))
            except:
                print (colored("\t\t\tERROR",'red'))
        
        elif action == 5:
            pass
        
        elif action == 0:
            #EXIT
            break
        
    print("hello :D")
    image_path = '../dataset/Mag.jpg'
    # facial_landmarks.draw_landmarks(image_path)
    # facial_landmarks.draw_parts(image_path)

    # load a dataset
    # 1) get key points (mylistdir workaround for OS DS_store)
    # facial_landmarks.store_key_points(auxilary.path_to_maindata)
    # 2) get differences
    # get_diff.get_diff(auxilary.path_to_csv_key_points, auxilary.path_to_maindata)\
    
    # print ("Loading the detector and predictor...\n")
    predictor , detector = facial_landmarks.load_pred_detec()
    # print ("Training the Classifier...\n")
    clf = SVM.svm_compare()
    # print ("Testing random image...\n")
    show_tests.show_tests(auxilary.path_to_maindata , clf, detector,predictor)
