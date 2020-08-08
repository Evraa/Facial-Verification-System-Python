from imutils import face_utils
import numpy as np
import argparse, os
import imutils
import dlib
import copy
import cv2
from auxilary import path_to_shape_predictor, shape_to_np, dominant_key_points, fixed_key_point, \
    how_sure, store_keys, create_key_points_data_frame
from collections import OrderedDict
from PIL import Image
from matplotlib import image
from matplotlib import pyplot
import delaunay

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


def draw_landmarks(image,shape,rect):
    '''
        To show red dot circle where key points exist.

        `image_path` is the path to the image

        `circle_type` dominant or no_dominant    
    '''
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

    delaunay.get_delaunay_points(shape,orig_image,returned = False)
    
    

    


def get_key_points(image_path, detector, predictor):
    img = Image.open(image_path)
    # image = cv2.imread(image_path)
    image = np.asarray(img).astype(np.uint8)
    rects = detector(image, 1)
    if len(rects) == 0:
        return None, False
    rect = rects[0]
    shape = predictor(image, rect)
    shape = shape_to_np(shape)
    return shape,True
