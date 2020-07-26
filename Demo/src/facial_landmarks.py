from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from auxilary import *
import timeit

def load_pred_detec(path_to_shape_predictor):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_to_shape_predictor)    
    return predictor , detector

def predict_shapes(image_path):
    predictor , detector = load_pred_detec(path_to_shape_predictor)
    #read the image
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    rect = rects[0]
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)
    return shape, rect, image


def draw_landmarks(image_path, circle_type = "no_dominant"):
    '''
        To show red dot circle where key points exist.

        `image_path` is the path to the image

        `circle_type` dominant or no_dominant    
    '''
    shape, rect, image = predict_shapes(image_path)
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # show the face number
    cv2.putText(image, "Face #{}".format(1), (x - 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    for (x, y) in shape:
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

    if circle_type == 'dominant':
        shape_dom = shape[dominant_key_points]
        for (x, y) in shape_dom:
            cv2.circle(image, (x, y), 4, (0, 255,0), -1)
        cv2.circle(image, (shape[fixed_key_point][0], shape[fixed_key_point][1]), 4, (255,0,0), -1)
    cv2.imshow(image_path, image)
    cv2.waitKey(0)

    



def store_key_points(image_set_paths):
    #prepare the predictor model
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_to_shape_predictor)
    folders = os.listdir(image_set_paths)
    for folder in folders:
        set_number = folder
        folder_path = image_set_paths+folder+"/"
        images = (os.listdir(folder_path))
        for im in images:
            print (f'image {im} from set {folder}')
            image_path = folder_path + im
            #read the image
            image = cv2.imread(image_path)
            image = imutils.resize(image, width=500)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale image
            rects = detector(gray, 1)
            # loop over the face detections
            if len (rects) == 0:
                print (f"image: {im} doesn't have faces!")
            for (i, rect) in enumerate(rects):
                shape = predictor(gray, rect)
                shape = shape_to_np(shape)
                store_keys(im,shape,set_number)