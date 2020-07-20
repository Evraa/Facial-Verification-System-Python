import numpy as np
import pandas as pd
import cv2
import os




def read_data(folder_path = '../dataset/'):
    files = os.listdir(folder_path)
    i = 0
    for img_file in files:
        img = cv2.imread(folder_path+img_file,0)
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()








if __name__ == "__main__":
    read_data()