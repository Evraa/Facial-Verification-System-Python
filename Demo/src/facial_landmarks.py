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
import face_recognition
from matplotlib import pyplot
import delaunay
from scipy import ndimage
from glob import glob
import pandas as pd

def load_pred_detec():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_to_shape_predictor)
    return predictor, detector


def get_shape(image_path, predictor, detector):
    #read the image
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=500)
    # image = face_recognition.(image)
    gray = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
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
    # i = 0
    if manual:
        for line in lines:
            # if i%2 == 0:
            #     color = (20,255,212)
            # else:
            #     color = (255,20,212)
            # i += 1
            rand_int_1 = int(np.random.random_integers(0,255))
            rand_int_2 = int(np.random.random_integers(0,255))
            rand_int_3 = int(np.random.random_integers(0,255))
            color = (rand_int_1, rand_int_2, rand_int_3)
            cv2.line(image, (line[0][0],line[0][1]), (line[1][0],line[1][1]), color, 3)

        cv2.imshow("Features", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        image = orig_image
        return orig_image
    else:
        delaunay.get_delaunay_points(shape,image,returned = False)
        image = orig_image
        return orig_image


def get_ratios(shape, image):
    '''
        Retutns a list of features (ratios):
        + width:        0-16
        + 0 L eyebrow   17-21
        + 1 R eyebrow   22-26
        + 2 L eye       36-39
        + 3 R eye       42-45
        + 4 Mouse       48-54
        + 5 Nose_width  31-35
        + 6 teeth_width  4-12
        + 7 eyebrows_wid 21-22
        + 8 eyes_width  39-42

        + Height:           27-8
        + 9 Nose_height     27-33
        + 10 L nose proj     30-31
        + 11 R nose proj     30-35
        + 12 M nose proj     30-33
        + 13 lips height    62-66
        + 14 Chin height     57-8
        + 15 L eye hight     37-41 // 38-40
        + 16 R eye hight     43-47 // 44-46
        + 17 Nose to mouth   33-51
        + 18 Mouse height    51-57

        + In addition to these ratios, we can add COLORS:
            - 11 Eyebrow color
            - 12 eye color
            - 13 skin color
    '''
    

    ratios = []
    lines = []
    width = distance_two_points(shape[0], shape[16])
    height = distance_two_points(shape[27], shape[8])
    
    ratios.append(distance_two_points(shape[17], shape[21])/width) ##This one is not affecting at all !!
    ratios.append(distance_two_points(shape[22], shape[26])/width) 
    ratios.append(distance_two_points(shape[36], shape[39])/width)
    ratios.append(distance_two_points(shape[42], shape[45])/width)
    ratios.append(distance_two_points(shape[48], shape[54])/width)
    ratios.append(distance_two_points(shape[31], shape[35])/width)
    ratios.append(distance_two_points(shape[4], shape[12])/width)
    ratios.append(distance_two_points(shape[21], shape[22])/width)
    ratios.append(distance_two_points(shape[39], shape[42])/width)

    lines.append([shape[17], shape[21]])
    lines.append([shape[22], shape[26]])
    lines.append([shape[36], shape[39]])
    lines.append([shape[42], shape[45]])
    lines.append([shape[48], shape[54]])
    lines.append([shape[31], shape[35]])
    lines.append([shape[4], shape[12]])
    lines.append([shape[21], shape[22]])
    lines.append([shape[39], shape[42]])

    ratios.append(distance_two_points(shape[27], shape[33])/height)
    ratios.append(distance_two_points(shape[30], shape[31])/height)
    ratios.append(distance_two_points(shape[30], shape[35])/height)
    ratios.append(distance_two_points(shape[30], shape[33])/height)
    ratios.append(distance_two_points(shape[62], shape[66])/height)
    ratios.append(distance_two_points(shape[57], shape[8])/height)
    ratios.append(distance_two_points(shape[33], shape[51])/height)
    ratios.append(distance_two_points(shape[51], shape[57])/height)
    
    # Left eye height estimate
    l1 = distance_two_points(shape[37], shape[41])
    l2 = distance_two_points(shape[40], shape[38])
    l  = l1+l2/2
    ratios.append(l/height)
    #RIGHT
    l1 = distance_two_points(shape[43], shape[47])
    l2 = distance_two_points(shape[44], shape[46])
    l  = l1+l2/2
    ratios.append(l/height)
    
    lines.append([shape[27], shape[33]])
    lines.append([shape[30], shape[31]])
    lines.append([shape[30], shape[35]])
    lines.append([shape[30], shape[33]])
    lines.append([shape[62], shape[66]])
    lines.append([shape[57], shape[8]])
    lines.append([shape[33], shape[51]])
    lines.append([shape[51], shape[57]])
    
    #eyebrows points 17-18-19-20-21-22-23-24-25-26
    eyebrows = shape[17:27]
    eyebrows_values = []
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for point in eyebrows:
        try:
            eyebrows_values.append(gray_image[point[0],point[1]])
        except IndexError:
            continue
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

    return lines, ratios

def extract_features(path,pred, detc, preview = False):
    '''
        For each image in the data set:
        + Fetch the name
        + Extract its key points
        + Calc lengths and all the features
        + Store them in an csv file `embedded_manual.csv`
    '''

    human_files = np.array(glob(path))
    embedded = np.zeros([len(human_files) ,22])
    labels = []
    for i, image_path in enumerate(human_files):
        if i%100 == 0:
            print (f'image: {i}')
        # Fetch the name
        human_name = image_path.split("/")[-1]
        # Do we have a face?
        state, shape, rect, image = get_shape(image_path, pred, detc)
        if not state:
            print (f"Error: this file: {image_path} doesn't have a face to detect!")
            continue
        
        _, features = get_ratios(shape, image)
        embedded[i] = features
        labels.append(human_name)

    df = pd.DataFrame(embedded)
    df["output"] = labels
    df.to_csv("../csv_files/embedded_3.csv",index=False)
    
    
def test_preview(blur = False, dataset_path = "../dataset/main_data/*/*", pred=None, detc=None, image_path = None):
    if not image_path:
        print("No path given")
        human_files = np.array(glob(dataset_path))
        image_count = len(human_files)
        rand_int = np.random.random_integers(0,image_count)
        image_path = human_files[rand_int]
    face_name = image_path.split("/")[-1]
    state, shape, rect, image = get_shape(image_path, pred, detc)
    while not state:
        print (f"Error: this file: {image_path} doesn't have a face to detect!")
        rand_int = np.random.random_integers(0,image_count)
        state, shape, rect, image = get_shape(image_path, pred, detc)

    lines, embeddings = get_ratios(shape, image)
    draw_landmarks(image,shape,rect, blur = blur, manual = True, lines = lines)
    
    return embeddings, face_name, image_path

def test_frame(frame, pred, detc ):
    '''
        + Takes frame from video
        + Returns that frame with a box around the faces detected
        + with each box a name and percentage of prediction
    '''
    state, shape, rect, image = get_shape_from_image (frame, pred, detc)
    if not state:
        return False, None, None
    
    return True, shape, rect

def get_shape_from_image (image, predictor, detector):
    #read the image
    # image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    if len(rects) == 0:
        return False,None,None,None
    rect = rects[0]
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)
    return True,shape, rect, image