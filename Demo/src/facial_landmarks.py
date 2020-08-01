from imutils import face_utils
import numpy as np
import argparse, os
import imutils
import dlib
import cv2
from auxilary import path_to_shape_predictor,shape_to_np,dominant_key_points,fixed_key_point,\
    how_sure, store_keys, create_key_points_data_frame
from collections import OrderedDict 
from PIL import Image
from matplotlib import image
from matplotlib import pyplot

def load_pred_detec():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_to_shape_predictor)    
    return predictor , detector

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
    
    cv2.imshow("Detected Face", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for (x, y) in shape:
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow("Predicted Facial points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if circle_type == 'dominant':
        shape_dom = shape[dominant_key_points]
        for (x, y) in shape_dom:
            cv2.circle(image, (x, y), 4, (0, 255,0), -1)
        cv2.circle(image, (shape[fixed_key_point][0], shape[fixed_key_point][1]), 4, (255,0,0), -1)

    cv2.imshow("Dominant Features", image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    
def draw_parts(image_path):
    shape, rect, image = predict_shapes(image_path)
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36))
    ])

    overlay = image.copy()
    output = image.copy()
    colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),\
        (168, 100, 168), (158, 163, 32),(163, 38, 32)]

    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]
        #how sure are we?
        percentage = how_sure (pts,name)
        name = name + " "+str(percentage) + " %"
        # check if are supposed to draw the jawline
        hull = cv2.convexHull(pts)
        cv2.drawContours(overlay, [hull], -1, colors[i], -1)
        # for parts extraction
        clone = image.copy()
        cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for (x, y) in pts:
            cv2.circle(clone, (x, y), 3, (0, 0, 255), -1)
        
        (x, y, w, h) = cv2.boundingRect(np.array(pts))
        roi = image[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        # show the particular face part
        cv2.imshow("ROI", roi)
        cv2.imshow("Image", clone)
        cv2.waitKey(0)
        
    alpha=0.75
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    cv2.imshow(image_path, output)
    cv2.waitKey(0)
    return 


def store_key_points(image_set_paths):
    #prepare the predictor model
    create_key_points_data_frame()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_to_shape_predictor)
    folders = os.listdir(image_set_paths)
    for folder in folders:
        set_number = folder
        folder_path = image_set_paths+folder+"/"
        images = (os.listdir(folder_path))
        for im in images:
            # print (f'image {im} from set {folder}')
            image_path = folder_path + im
            # print (image_path)
            #read the image
            img = Image.open(image_path)
            # print (image.size)
            # print (image.mode)
            image = np.asarray(img)
            # image = cv2.imread(image_path,0)
            # print (image.shape)
            # image = imutils.resize(image, width=500)
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale image
            # cv2.imshow("i",image)
            
            rects = detector(image, 1)
            # loop over the face detections
            if len (rects) == 0:
                print (f"image: {im} doesn't have faces!")
            for (_, rect) in enumerate(rects):
                shape = predictor(image, rect)
                shape = shape_to_np(shape)
                # for (x, y) in shape:
                #     cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
                # pyplot.imshow(image)
                # pyplot.show()
                # input('ev')
                # print (im)
                iamge_name = str(im)
                store_keys(iamge_name,shape,set_number)

def get_key_points(image_path,detector,predictor):
    img = Image.open(image_path)
    # image = cv2.imread(image_path)
    image = np.asarray(img)
    # print (type(image))
    rects = detector(image, 1)
    if len(rects) == 0:
        return None,False
    rect = rects[0]
    shape = predictor(image, rect)
    shape = shape_to_np(shape)
    return shape,True
