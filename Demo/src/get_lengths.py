from auxilary import path_to_shape_predictor , shape_to_np, calc_distances, path_to_csv_lengths, create_demo,\
    path_to_all_dataset
from imutils import face_utils
import numpy as np
import argparse, os
import imutils
import dlib
import cv2

def get_images_lengths():

    create_demo(fileName=path_to_csv_lengths) 
    path_to_images = path_to_all_dataset
    
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
        for (_, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            #I am assuming only one face is represented in this image
            calc_distances(image_name,shape)