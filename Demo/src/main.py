#GLOBAL IMPORTS
import numpy as np
from termcolor import colored
from glob import glob

#LOCAL IMPORTS
import auxilary
import facial_landmarks
import SVM
import show_tests
import NN
from create_model import create_model
import plot_weights
import video_capture

#MAGGIE COMMENT THESE TWO
import face_recognition
import show_results 


def take_action():
    print (colored("\t\t\tWelcome to our humble applicaion", 'yellow'))
    print (colored("\tBefore we start, please make sure that your dataset is placed at Demo/dataset/*",'red'))
    print (colored("\tand the two models are placed at Demo/src/",'red'))

    print (colored("\tTo Transform your data to affine dataset, \t\tpress 1 (Do ONCE)",'cyan'))
    print (colored("\tTo Create Embeddings for the data you just transformed, press 2 (Do ONCE)",'cyan'))
    print (colored("\tTo Test classification using Euclidean equation, \tpress 3",'cyan'))
    print (colored("\tTo Train the NN model for classification, \t\tpress 4 (Do ONCE)",'cyan'))
    print (colored("\tTo Test classification using Neural Networks, \t\tpress 5",'cyan'))
    print (colored("\tTo Test classification using SVM with Bayes, \t\tpress 6",'cyan'))
    print (colored("\tTo Test classification using NN with Blured images, \tpress 7",'cyan'))
    print (colored("\tTo Extract Features Manually from faces and store them, press 8 (Do ONCE)",'cyan'))
    print (colored("\tTo Test classification using Neural Networks 2, \tpress 9",'cyan'))
    print (colored("\tfor Mag! to get inputs and targets \t\t\tpress 10",'cyan'))
    print (colored("\tTo Calc new weights and plot them \t\t\tpress 11 (Do ONCE)",'cyan'))
    print (colored("\tTo Test video using Neural Networks 2, \t\t\tpress 12",'cyan'))
    print (colored("\tTo Test classification using Neural Networks, \t\tpress 13",'cyan'))
    print (colored("\tTo Test classification using Neural Networks 2, \tpress 14",'cyan'))
    

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
                show_results.Euc_result_preview()
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
            print (colored("\t\t\tSTARTING",'green'))            
            try:
                show_results.NN_result_preview()
                print (colored('\t\t\tDONE','green'))
            except:
                print (colored("\t\t\tERROR",'red'))
    
        elif action == 6:
            print (colored("\t\t\tTraining SVM clf",'green'))            
            try:
                clf = SVM.svm_compare()
                print (colored('\t\t\tSuccessfully trained clf','green'))
            except:
                print (colored("\t\t\tERROR in training",'red'))
            
            print (colored("\t\t\Showing Results",'green'))            
            # try:
            show_tests.show_tests(auxilary.path_to_maindata , clf, detc,pred)
            print (colored('\t\t\tDone','green'))
            # except:
            #     print (colored("\t\t\tERROR",'red'))

        elif action == 7:
            print (colored("\t\t\tSTARTING",'green'))            
            try:
                show_results.NN_result_preview(blur=True)
                print (colored('\t\t\tDONE','green'))
            except:
                print (colored("\t\t\tERROR",'red'))
        elif action == 8:
            print (colored("\t\t\tSTARTING",'green'))    
            try:
                dataset_path = '../dataset/main_data/*/*'
                facial_landmarks.extract_features(path = dataset_path,pred=pred, detc=detc)
                print (colored('\t\t\tDONE','green'))
            except:
                print (colored("\t\t\tERROR",'red'))
        elif action == 9:
            print (colored("\t\t\tSTARTING",'green'))            
            try:
                show_results.NN_result_preview(second=True,pred=pred, detc=detc)
                print (colored('\t\t\tDONE','green'))
            except:
                print (colored("\t\t\tERROR",'red'))
        elif action == 10:
            print (colored("\t\t\tSTARTING",'green'))            
            try:
                inputs, targets, encoded_names = NN.prepare_data()
                image_count = len(targets)
                random_int = np.random.random_integers(0,image_count)
                
                print (colored("first row's features: ", "yellow"))
                print (inputs[random_int])
                print (colored("First row's target value:", "yellow"))
                print (targets[random_int])
                print (colored("what is the index of the one ?", "yellow"))
                one = list(np.where(targets[random_int] == 1))
                print (one)
                print (colored("encoded value:","yellow"))
                print (encoded_names.inverse_transform( one ))
                print (colored('\t\t\tDONE','green'))
            except:
                print (colored("\t\t\tERROR",'red'))

        elif action == 11:
            print (colored("\t\t\tSTARTING",'green'))            
            try:
                plot_weights.run_main(csv_path = "../csv_files/embedded_2.csv")
                print (colored('\t\t\tDONE','green'))
            except:
                print (colored("\t\t\tERROR",'red'))
        
        elif action == 12:
            print (colored("\t\t\tSTARTING",'green'))            
            try:
                video_capture.main_loop(pred, detc)
                print (colored('\t\t\tDONE','green'))
            except:
                print (colored("\t\t\tERROR",'red'))
        
        elif action == 13:
            print (colored("\t\t\tSTARTING",'green'))            
            try:
                show_results.NN_result_preview(pos_fals = True, blur=True)
                print (colored('\t\t\tDONE','green'))
            except:
                print (colored("\t\t\tERROR",'red'))
        
        elif action == 14:
            print (colored("\t\t\tSTARTING",'green'))            
            try:
                show_results.NN_result_preview(second = True, pos_fals = True, pred=pred, detc=detc)
                print (colored('\t\t\tDONE','green'))
            except:
                print (colored("\t\t\tERROR",'red'))

        elif action == 0:
            #EXIT
            break
        
    
