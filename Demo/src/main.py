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

#MAGGIE COMMENT THESE TWO
import face_recognition
import show_results 


def take_action():
    print (colored("\t\t\tWelcome to our application", 'yellow'))
    print (colored("\tBefore we start, please make sure that your dataset is placed at Demo/dataset/*",'red'))
    print (colored("\tand the two sequential NN models are placed at Demo/src/",'red'))

    print (colored("\tTo Transform and embed your data to affine dataset, \t\tpress 1 (Do ONCE)",'cyan'))
    print (colored("\tTo Extract Features Manually from faces and store them, press 2 (Do ONCE)",'cyan'))
    print (colored("\tTo Train the SVM CLF, \tpress 3",'cyan'))
    print (colored("\tTo Train the NN model for classification, \t\tpress 4 (Do ONCE)",'cyan'))
    print (colored("\tTo Test classification using Neural Networks, \t\tpress 5",'cyan'))
    print (colored("\tTo Test classification using SVM with Bayes, \t\tpress 6",'cyan'))
    print (colored("\tTo Test classification using Euclidean equation, \tpress 7",'cyan'))
    print (colored("\tTo Calc new weights and plot them \t\t\tpress 10 (Do ONCE)",'cyan'))
    print (colored("\tBucket decisions\t\t\tpress 11",'cyan'))
    

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
    file_dataset = '../dataset/main_data/*/*'
    embedded_dataset = '../csv_files/embedded_2.csv'

    while True:
        action = take_action()

        if action == 1:
            #Affine transformation
            print (colored("\t\t\tSTARTING",'green'))
            dataset_path = '../dataset/main_data/*/*'
            human_files = np.array(glob(dataset_path))
            face_recognition.affine_transformation(human_files, pred, detc)

            # embedding time
            dataset_path = '../dataset/lfw_affine/*/*'
            human_files = np.array(glob(dataset_path))
            model = create_model()
            model.load_weights('../open_face.h5')
            face_recognition.store_embeddings(human_files, model)

            try:
                # face_recognition.affine_transformation(human_files,pred,detc)
                print (colored('\t\t\tDONE','green'))
            except:
                print (colored("\t\t\tERROR",'red'))

        elif action == 2:
            print (colored("\t\t\tSTARTING",'green'))
            dataset_path = '../dataset/main_data/*/*'
            facial_landmarks.extract_features(path=dataset_path, pred=pred, detc=detc)
            # try:
            #     dataset_path = '../dataset/main_data/*/*'
            #     facial_landmarks.extract_features(path = dataset_path,pred=pred, detc=detc)
            #     print (colored('\t\t\tDONE','green'))
            # except:
            #     print (colored("\t\t\tERROR",'red'))
        elif action == 3:
            print(colored("\t\t\tSTARTING", 'green'))
            try:
                path = '../csv_files/csv_differences_prof.csv'
                clf = SVM.svm_compare(path)
                print(colored('\t\t\tDONE', 'green'))
            except:
                print(colored("\t\t\tERROR", 'red'))
        
        elif action == 4:
            dataset_path = '../csv_files/embedded_2.csv'
            print (colored("\t\t\tSTARTING",'green'))            
            try:
                NN.train(dataset_path)
                print (colored('\t\t\tDONE','green'))
            except:
                print (colored("\t\t\tERROR",'red'))
        
        elif action == 5:
            dataset_path = '../csv_files/embedded_2.csv'
            print(colored("\t\t\tSTARTING", 'green'))
            show_results.NN_result_preview(dataset_path, detc=detc, pred=pred)
            print(colored('\t\t\tDONE', 'green'))
            # try:
            #     dataset_path = '../csv_files/embedded_2.csv'
            #     bool = input(colored("\t\t\tBlurred? (Y/N): ",'blue'))
            #     if bool == "Y":
            #         bool = True
            #     else:
            #         bool = False
            #     print (colored("\t\t\tSTARTING",'green'))
            #     show_results.NN_result_preview(dataset_path, second=bool, detc=detc, pred=pred)
            #     print (colored('\t\t\tDONE','green'))
            # except:
            #     print (colored("\t\t\tERROR",'red'))
    
        elif action == 6:
            print (colored("\t\t\tTraining SVM clf",'green'))
            try:
                # path = '../csv_files/csv_differences_prof.csv'
                # clf = SVM.svm_compare(path)
                print (colored('\t\t\tSuccessfully trained clf','green'))
            except:
                print (colored("\t\t\tERROR in training",'red'))
            print (colored("\t\tShowing Results",'green'))
            show_tests.show_tests(auxilary.path_to_maindata, detc, pred, False)
            # try:
            #     show_tests.show_tests(auxilary.path_to_maindata , clf, detc,pred)
            #     print (colored('\t\t\tDone','green'))
            # except:
            #     print (colored("\t\t\tERROR",'red'))

        elif action == 7:
            print(colored("\t\t\tSTARTING", 'green'))
            try:
                dataset_path = '../csv_files/embedded.csv'
                show_results.Euc_result_preview(dataset_path)
                print(colored('\t\t\tDONE', 'green'))
            except:
                print(colored("\t\t\tERROR", 'red'))

        elif action == 10:
            print (colored("\t\t\tSTARTING",'green'))            
            try:
                plot_weights.run_main(csv_path = "../csv_files/embedded_2.csv")
                print (colored('\t\t\tDONE','green'))
            except:
                print (colored("\t\t\tERROR",'red'))
        elif action == 11:
            dataset_path = '../dataset/main_data/'
            img = input(colored("\t\t\tImage path: ",'blue'))
            show_tests.show_tests(dataset_path, detc, pred, True, img, embedded_dataset)
        elif action == 0:
            #EXIT
            break
        
    
