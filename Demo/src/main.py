from auxilary import *
from facial_landmarks import *
from get_lengths import *
from identify_faces import *


if __name__ == "__main__":
    print ("hello :D")
    #TODO: add a pretty CLI for users
    
    image_files_path = "../dataset/"
    image_files = os.listdir(image_files_path)
    for image_file in image_files:
        show_landmarks(image_path=image_files_path+image_file)
