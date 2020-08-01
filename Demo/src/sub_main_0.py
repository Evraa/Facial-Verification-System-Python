import auxilary
import face_recognition
import facial_landmarks
import numpy as np
from glob import glob


if __name__ == "__main__":
    print ("hello :D")
    #Load the predictor and detector only once
    pred , detec = facial_landmarks.load_pred_detec()
    
    #Load the dataset
    human_files = np.array(glob("../dataset/lfw/*/*"))
    face_recognition.affine_transformation(human_files,pred,detec)
    print ("DONE :D")