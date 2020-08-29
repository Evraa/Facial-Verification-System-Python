from numpy import average

from auxilary import path_to_shape_predictor, shape_to_np, calc_distances, path_to_csv_lengths, create_demo, \
    path_to_all_dataset, mylistdir, distance_two_points, add_row
import numpy as np
import argparse, os
import pandas as pd
import imutils
import dlib
import cv2


def create_ratio_frame():
    '''
        Creates a csv file with the essential feature data
    '''
    my_dict = {'image_set': [],
               'image_name': [],

               # eyebrow values
               'left_eyebr_width': [],
               'right_eyebr_width': [],
               'left_eyebr_y_pos': [],
               'right_eyebr_y_pos': [],
               'dist_inner_eyebr': [],
               'dist_outer_eyebr': [],
               'eyebr_inner_slope': [],
               'eyebr_outer_slope': [],

               # eye values
               'dist_inner_eyes': [],
               'dist_outer_eyes': [],
               'eye_shape': [],

               # nose values
               'nose_width': [],
               'nose_height': [],
               'nose_shape': [],

               #mouth values
               'mouth_width': [],
               'mouth_height': [],

               # face shape
               'face_width1': [],
               'face_width2': [],
               'face_width3': [],

               #facial ratios
               'eye_to_eyebr': [],
               'nose_to_eyebr': [],
               'philtrum': []
               }

    fileName = '../csv_files/facial_ratios.csv'
    # # convert it into dataframe
    df = pd.DataFrame(my_dict)
    # # transfer into csv file
    df.to_csv(fileName, index=False)

    get_image_ratios(my_dict)


def get_image_ratios(my_dict):
    create_demo(fileName=path_to_csv_lengths)
    path_to_images = '../dataset/grouped/'

    directories = mylistdir(path_to_images)
    files = []
    for directory in directories:
        images = mylistdir(path_to_images + directory)
        for image in images:
            str = path_to_images + directory + "/" + image
            files.append(str)

    for image_path in files:
        print(image_path)
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
            # calc_distances(image_path,shape)
            dicts = append_ratios(image_path, shape, my_dict)
    df = pd.DataFrame.from_dict(my_dict)
    df.to_csv('../csv_files/facial_ratios.csv', index=False, header=True)
    print(df)


def append_ratios(image_path, shape, dicts):
    split_dir = image_path.split('/')
    name = split_dir[4]
    set = split_dir[3]
    my_dict = dicts

    width1 = distance_two_points(shape[0], shape[16])
    width2 = distance_two_points(shape[2], shape[14])
    width3 = distance_two_points(shape[5], shape[11])
    height = distance_two_points((shape[19] + shape[24])/2, shape[8])
    chin = shape[8]

    left_eyebrow_outer = shape[17]
    left_eyebrow_inner = shape[21]
    left_eyebrow_center = shape[19]
    right_eyebrow_outer = shape[26]
    right_eyebrow_inner = shape[22]
    right_eyebrow_center = shape[24]

    left_eye_outer = shape[36]
    left_eye_inner = shape[39]
    right_eye_outer = shape[45]
    right_eye_inner = shape[42]
    left_eye_slope = left_eye_outer[1] - shape[37][1] / left_eye_outer[0] - shape[37][0]
    right_eye_slope = right_eye_outer[1] - shape[44][1] / right_eye_outer[0] - shape[44][0]

    nose_left = shape[31]
    nose_right = shape[35]
    nose_bottom = shape[33]
    nose_top = shape[27]

    mouth_left = shape[48]
    mouth_right = shape[54]
    mouth_top = shape[51]
    mouth_bottom = shape[57]
    lip = shape[66]

    my_dict['image_name'].append(name)
    my_dict['image_set'].append(set)
    my_dict['left_eyebr_width'].append(distance_two_points(left_eyebrow_outer, left_eyebrow_inner) / width2)
    my_dict['right_eyebr_width'].append(distance_two_points(right_eyebrow_inner, right_eyebrow_outer) / width2)
    my_dict['left_eyebr_y_pos'].append(distance_two_points(left_eyebrow_center, chin) / height)
    my_dict['right_eyebr_y_pos'].append(distance_two_points(right_eyebrow_center, chin) / height)
    my_dict['dist_inner_eyebr'].append(distance_two_points(left_eyebrow_inner, right_eyebrow_inner) / width2)
    my_dict['dist_outer_eyebr'].append(distance_two_points(left_eyebrow_outer, right_eyebrow_outer) / width2)
    left_inner_slope = (left_eyebrow_center[1] - left_eyebrow_inner[1]) / (
            left_eyebrow_center[0] - left_eyebrow_inner[0])
    right_inner_slope = (right_eyebrow_center[1] - right_eyebrow_inner[1]) / (
                right_eyebrow_center[0] - right_eyebrow_inner[0])
    left_outer_slope = (left_eyebrow_outer[1] - left_eyebrow_center[1]) / (
                left_eyebrow_outer[0] - left_eyebrow_center[0])
    right_outer_slope = (left_eyebrow_outer[1] - left_eyebrow_center[1]) / (
                left_eyebrow_outer[0] - left_eyebrow_center[0])
    my_dict['eyebr_inner_slope'].append((left_inner_slope + right_inner_slope) / 2)
    my_dict['eyebr_outer_slope'].append((left_outer_slope + right_outer_slope) / 2)
    my_dict['dist_inner_eyes'].append(distance_two_points(left_eye_inner, right_eye_inner) / width2)
    my_dict['dist_outer_eyes'].append(distance_two_points(left_eye_outer, right_eye_outer) / width2)
    my_dict['eye_shape'].append((right_eye_slope + left_eye_slope) / 2)
    nheight = distance_two_points(nose_left, nose_right)
    nwidth = distance_two_points(nose_bottom, nose_top)
    my_dict['nose_width'].append(nheight / width2)
    my_dict['nose_height'].append(nwidth / height)
    my_dict['nose_shape'].append(nwidth / nheight)
    my_dict['mouth_width'].append(distance_two_points(mouth_left, mouth_right) / width2)
    my_dict['mouth_height'].append(distance_two_points(mouth_top, mouth_bottom) / height)
    my_dict['face_width1'].append(width1)
    my_dict['face_width2'].append(width2)
    my_dict['face_width3'].append(width3)
    my_dict['eye_to_eyebr'].append((distance_two_points(shape[38],shape[20]) + distance_two_points(shape[43],shape[23]))/ 2)
    my_dict['nose_to_eyebr'].append((distance_two_points(nose_bottom,left_eyebrow_inner) + distance_two_points(nose_bottom,right_eyebrow_inner)) / 2)
    my_dict['philtrum'].append(distance_two_points(nose_bottom, mouth_bottom) / height)
    return my_dict

create_ratio_frame()