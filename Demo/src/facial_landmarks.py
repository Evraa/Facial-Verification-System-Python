from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from auxilary import *


def distance_two_points(x,y):
    x_diff = x[0] - y[0]
    x_pow = x_diff**2
    y_diff = x[1] - y[1]
    y_pow = y_diff**2
    return np.sqrt(x_pow + y_pow)

def calc_distances(image_name,shape):
    '''
        Calculate the distances:
            + Left eyebrow
            + Right eyebrow
            + Left eye socket
            + Right eye socket
            + Left nostril
            + Right nostril
            + Mouse
        
        then append them in the dictionary
    '''
    print (f'this is image: {image_name}')
    face_features = {   'image_name'    :[],
                        #7 major points
                        'Eye_br_L'      :[],
                        'Eye_br_R'      :[],
                        'Eye_soc_L'     :[],
                        'Eye_soc_R'     :[],
                        'Nostril_L'     :[],
                        'Nostril_R'     :[],
                        'Moustache'     :[],
                        'Face_Width'    :[],
                        'Face_Height'   :[]
                        }

    face_features['image_name'].append(image_name)
    # eyebrows
    face_features['Eye_br_L'].append(distance_two_points(shape[17],shape[21]))
    face_features['Eye_br_R'].append( distance_two_points(shape[22],shape[26]))
    #eye sockets
    face_features['Eye_soc_L'].append(distance_two_points(shape[36],shape[39]))
    face_features['Eye_soc_R'].append(distance_two_points(shape[42],shape[45]))
    #nostrils
    face_features['Nostril_L'].append(distance_two_points(shape[31],shape[32]))
    face_features['Nostril_R'].append(distance_two_points(shape[34],shape[35]))
    #Mouse
    face_features['Moustache'].append(distance_two_points(shape[48],shape[54]))
    #width
    face_features['Face_Width'].append(distance_two_points(shape[0],shape[16]))
    #height
    x = shape[19]
    y = shape[24]
    mid_point = ((x[0]+y[0])/2 ,(x[1]+y[1])/2 )
    face_features['Face_Height'] = distance_two_points(mid_point,shape[8])
    
    #store the data
    df = read_csv()
    add_row(df,face_features)



def detect_face():

    path_to_images = "../dataset/"
    path_to_shape_predictor = "../shape_predictor_68_face_landmarks.dat"

    files = os.listdir(path_to_images)
    
    for image_name in files:
        image_path = path_to_images + image_name
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(path_to_shape_predictor)

        # load the input image, resize it, and convert it to grayscale
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale image
        rects = detector(gray, 1)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            #I am assuming only one face is represented in this image
            calc_distances(image_name,shape)
     
if __name__ == "__main__":
    create_demo()
    detect_face()