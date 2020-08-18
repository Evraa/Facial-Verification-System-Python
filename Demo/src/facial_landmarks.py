from imutils import face_utils
import numpy as np
import argparse, os
import imutils
import dlib
import copy
import cv2
from auxilary import path_to_shape_predictor, shape_to_np, dominant_key_points, fixed_key_point, \
    how_sure, store_keys, create_key_points_data_frame, mylistdir, distance_two_points
from collections import OrderedDict
from PIL import Image
from matplotlib import image
from matplotlib import pyplot
import delaunay
from scipy import ndimage
from glob import glob


def load_pred_detec():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_to_shape_predictor)
    return predictor, detector


def get_shape(image_path, predictor, detector):
    #read the image
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    if len(rects) == 0:
        return False,None,None,None
    rect = rects[0]
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)
    return True,shape, rect, image


def blur_image(image, sigma):
    gausBlur = cv2.GaussianBlur(image, (sigma, sigma),0)  

    very_blurred = ndimage.gaussian_filter(image, sigma=5)
    cv2.imshow('Gaussian Blurring', very_blurred) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    return very_blurred

def draw_landmarks(image,shape,rect, blur = False, manual = False, lines = None):
    '''
        To show red dot circle where key points exist.

        `image_path` is the path to the image
    '''

    cv2.imshow("Original Image", image)
    cv2.waitKey(0)

    if blur:
        image = blur_image(image, 5)
    orig_image = copy.deepcopy(image)
    # shape, rect, image = predict_shapes(image_path)
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # show the face number
    cv2.putText(image, "Face #{}".format(1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detected Face", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for (x, y) in shape:
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow("Predicted Facial points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if manual:
        pass
    else:
        delaunay.get_delaunay_points(shape,image,returned = False)
        image = orig_image
        return orig_image


def get_ratios(shape, image):
    '''
        Retutns a list of features (ratios):
        + width:    0-16
        + L eyebrow 17-21
        + R eyebrow 22-26
        + L eye     36-39
        + R eye     42-46
        + Mouse     48-54
        + Nose_width    31-35


        + Height:       27-8
        + Nose_height   27-33
        + L nose proj   30-31
        + R nose proj   30-35
        + M nose proj   30-33
        + lips height   62-66

        + In addition to these ratios, we can add COLORS:
            - Eyebrow color
            - eye color
            - skin color
    '''
    # print (shape)
    # print (type(image))
    # print (image.shape)
    # print (image[0])
    # input ("ev")

    ratios = []

    width = distance_two_points(shape[0], shape[16])
    height = distance_two_points(shape[27], shape[8])

    ratios.append(distance_two_points(shape[17], shape[21])/width)
    ratios.append(distance_two_points(shape[22], shape[26])/width)
    ratios.append(distance_two_points(shape[36], shape[39])/width)
    ratios.append(distance_two_points(shape[42], shape[46])/width)
    ratios.append(distance_two_points(shape[48], shape[54])/width)
    ratios.append(distance_two_points(shape[31], shape[35])/width)

    ratios.append(distance_two_points(shape[27], shape[33])/height)
    ratios.append(distance_two_points(shape[30], shape[31])/height)
    ratios.append(distance_two_points(shape[30], shape[35])/height)
    ratios.append(distance_two_points(shape[30], shape[33])/height)
    ratios.append(distance_two_points(shape[62], shape[66])/height)
    
    #eyebrows points 17-18-19-20-21-22-23-24-25-26
    eyebrows = shape[17:27]
    eyebrows_values = []
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for point in eyebrows:
        eyebrows_values.append(gray_image[point[0],point[1]])
    ratios.append(np.average(eyebrows_values)/255)

    #nose color (SKIN)
    skin = shape[27:36]
    skin_values = []
    for point in skin:
        skin_values.append(gray_image[point[0],point[1]])
    ratios.append(np.average(skin_values)/255)

    #Lips color
    lips = shape[48:68]
    lips_values = []
    for point in lips:
        lips_values.append(gray_image[point[0],point[1]])
    ratios.append(np.average(lips_values)/255)

    return ratios

def extract_features(path,pred, detc, preview = False):
    '''
        For each image in the data set:
        + Fetch the name
        + Extract its key points
        + Calc lengths and all the features
        + Store them in an csv file `embedded_manual.csv`
    '''

    human_files = np.array(glob(path))
    
    for image_path in human_files:
        # Fetch the name
        human_name = image_path.split("/")[-1].split("\\")[1]
        
        # Do we have a face?
        state, shape, rect, image = get_shape(image_path, pred, detc)
        if not state:
            print (f"Error: this file: {image_path} doesn't have a face to detect!")
            continue
        
        ratios = get_ratios(shape, image)
        
        


