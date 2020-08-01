#Global imports
from glob import glob
import numpy as np
import os
import openface
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
    return file_name[0]

def move_images_with_no_predicted_faces(human_files,pred,detec):
    #Loop on these faces, and get the key points of each
    falsy = 0
    falsy_dir = '../dataset/falsy/'
    for i, human_file in enumerate(human_files):
        state, shape, rect, image = facial_landmarks.get_shape(human_file, pred, detec)
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


#Main function
def face_recognition(dataset_path = "../dataset/lfw/*/*"):
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
    #Affine transformation
    face_aligner = openface.AlignDlib(auxilary.path_to_shape_predictor)
    # Use openface to calculate and perform the face alignment
    # alignedFace = face_aligner.align(534, image, face_rect, \
    #     landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    


face_recognition()