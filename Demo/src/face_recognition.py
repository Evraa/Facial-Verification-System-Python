#Global imports
from glob import glob
import numpy as np
import os
import openface
import cv2
from create_model import create_model
import pandas as pd

#local imports
import facial_landmarks
import auxilary

#Global variables


#Internal methods
def get_name_from_path(path):
    '''
        + Path example ../dataset/lfw\Jacques_Chirac\Jacques_Chirac_0006.jpg
        + Result: Jacques_Chirac
    '''
    #split /
    file_name = path.split('/')
    #split \
    file_name = file_name[len(file_name)-1]
    file_name = path.split('\\')
    return file_name[1]

def get_folder_from_path(path):
    '''
        + Path example ../dataset/lfw\Jacques_Chirac\Jacques_Chirac_0006.jpg
        + Result: Jacques_Chirac\Jacques_Chirac_0006.jpg
    '''
    #split /
    file_name = path.split('/')
    #split \
    file_name = file_name[len(file_name)-1]
    file_name = path.split('\\')
    return file_name[1] + '\\' + file_name[2]


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

def move_images_with_no_predicted_faces(human_files,pred,detec):
    #Loop on these faces, and get the key points of each
    falsy = 0
    falsy_dir = '../dataset/falsy/'
    for i, human_file in enumerate(human_files):
        state, _, _, _ = facial_landmarks.get_shape(human_file, pred, detec)
        if not state:
            print (human_file)
            falsy += 1
            file_name = human_file.split('/')
            file_name = file_name[len(file_name)-1]
            file_name = file_name.split('\\')
            falsy_img_pth = falsy_dir + file_name[len(file_name)-1]
            os.replace(human_file, falsy_img_pth)
        if i%100 == 0:
            print (f'iteration {i} falsy: {falsy}')

def get_labesl(human_files):
    labels = []
    for human_file in human_files:
        labels.append(get_name_from_path(human_file))
    return labels

def affine_transformation(human_files,pred,detec,preview = False, image_num=0):
    '''
        For each Image in the dataset
            + Extract the key points
            + Do affine transformation
            + Store it
    '''
    #Affine transformation
    falsy_dir = '../dataset/falsy/'
    face_aligner = openface.AlignDlib(auxilary.path_to_shape_predictor)
    affine_dir = '../dataset/lfw_affine/'

    if preview:
        # for i ,human_file in enumerate(human_files):
        #     folders =  (get_folder_from_path(human_file))
        #     folders = folders.split('\\')
        #     if folders[0] == 'LeBron_James':
        #         print (i, human_file)
        # input("e")
        image_count = len(human_files) - 1
        rand_int = np.random.random_integers(0,image_count)
        # rand_int = image_num
        human_file = human_files[rand_int]
        while not os.path.exists(human_file):
            rand_int = np.random.random_integers(0,image_count)
            human_file = human_files[rand_int]
        state, shape, rect, image = facial_landmarks.get_shape(human_file, pred, detec)
        while not state:
            state, shape, rect, image = facial_landmarks.get_shape(human_file, pred, detec)

        facial_landmarks.draw_landmarks(image,shape,rect)
        alignedFace = face_aligner.align(96, image, rect, \
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        cv2.imshow("Aligned Face", alignedFace)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return alignedFace, get_name_from_path(human_file), human_file
    
    for i, human_file in enumerate(human_files):
        state, shape, rect, image = facial_landmarks.get_shape(human_file, pred, detec)
        if state:
            file_path = affine_dir + get_folder_from_path(human_file)
            create_folder(affine_dir + get_name_from_path(human_file))
            alignedFace = face_aligner.align(96, image, rect, \
                    landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            cv2.imwrite(file_path, alignedFace)
            # plt.imshow(face_aligned)
        else:
            print ('Sorry, that image does not have faces!')
            file_name = human_file.split('/')
            file_name = file_name[len(file_name)-1]
            file_name = file_name.split('\\')
            falsy_img_pth = falsy_dir + file_name[len(file_name)-1]
            os.replace(human_file, falsy_img_pth)
        if i%100 == 0:
            print (f'Working: .... iteration {i}')

def store_embeddings(human_files,model):
    labels = get_labesl(human_files)
    data_size = len(human_files)
    embedded = np.zeros((data_size, 128))
    for i, human_file in enumerate(human_files):
        try:    
            image = cv2.imread(human_file,1)
            image = (image / 255.).astype(np.float32)
            embedded[i] = model.predict(np.expand_dims(image, axis=0))[0]
            if i%100 == 0:
                print (f'Working: iteration {i}')
        except:
            print (f'Failed: {human_file}')

    #STORE
    df = pd.DataFrame(embedded)
    df["output"] = labels
    df.to_csv("../csv_files/embedded.csv",index=False)
    
#Main function
def face_recognition(dataset_path = "../dataset/lfw/*/*", preview=False, image_num = 0):
    '''
        + My Goal?
        + For each face in the dataset
            - Detect face
            - Extract landmarks
            - Rotate/Scale the image...Affine transformation
            - Extract the embeddings
            - Store them along side with the unique name of the person

    '''
    #Load the predictor and detector only once
    pred , detec = facial_landmarks.load_pred_detec()
    
    #Load the dataset
    human_files = np.array(glob(dataset_path))
    
    #Remove images that we are not able to detect faces on
    # move_images_with_no_predicted_faces(human_files,pred,detec)

    #Affine Transformation
    # affine_transformation(human_files,pred,detec)

    #Create model for 128 features extraction
    model = create_model()
    model.load_weights('../open_face.h5')
    # store_embeddings(human_files,model)

    if preview:
        # Show the image
        # show the image with key points
        # show the affine image
        # print out the embeddings
        image,face_name, human_file_path = affine_transformation(human_files,pred,detec,preview=preview, image_num = image_num)
        image = (image / 255.).astype(np.float32)
        embeddings = model.predict(np.expand_dims(image, axis=0))[0] 
        # print(model.predict(np.expand_dims(image, axis=0))[0])
        # dataset_names = []
        # for human_file in human_files:
        #     name = get_name_from_path(human_file)
        #     direc = "../dataset/lfw_affine/" + str(name) + "/"
        #     file_count = os.listdir(direc)
        #     if name not in dataset_names and len(file_count) > 0:
        #         dataset_names.append(name)
        print ("Read embeddings")
        
        return embeddings, face_name, human_file_path





# face_recognition(dataset_path = "../dataset/lfw_affine/*/*")