# from auxilary import path_to_all_dataset,os,path_to_yalefaces
# from facial_landmarks import draw_landmarks, draw_parts,store_key_points
import facial_landmarks
import auxilary
import os
import get_diff
import SVM
import show_tests
# from get_lengths import *
# from identify_faces import *
# from calc_weights import calc_weights
# from delaunay import get_delaunay_points,store_shape_tri

'''
    Jobs allowed:
        + show an image with facial landmarks on it (dominance option)
        + show an image with drawings on it
        + store key points
        + store lengths...done and wont need it anymore, but it's still there!
        + calc weights
        
'''




if __name__ == "__main__":
    print("hello :D")
    # image_path = '../dataset/Magag.jpg'
    # facial_landmarks.draw_landmarks(image_path)
    # facial_landmarks.draw_parts(image_path)

    # print ("Loading the detector and predictor...\n")
    predictor , detector = facial_landmarks.load_pred_detec(auxilary.path_to_shape_predictor)
    print ("Training the Classifier...\n")
    clf = SVM.svm_compare()
    print ("Testing random image...\n")
    show_tests.show_tests(auxilary.path_to_yalefaces,clf,detector,predictor)
