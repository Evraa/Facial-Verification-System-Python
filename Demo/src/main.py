from auxilary import *
from facial_landmarks import *
from get_lengths import *
from identify_faces import *


'''
    Jobs allowed:
        + show an image with facial landmarks on it (dominance option)
        + show an image with drawings on it
        + store key points
        + store lengths...done and wont need it anymore, but it's still there!
        + calc weights
        
'''

if __name__ == "__main__":
    print ("hello :D")
    #TODO: add a pretty CLI for users
    create_key_points_data_frame()
    store_key_points(path_to_images_grouped)
    # image_files_path = "../dataset/"
    # image_files = os.listdir(image_files_path)
    # for image_file in image_files:
    #     show_landmarks(image_path=image_files_path+image_file)
