from auxilary import path_to_all_dataset
from facial_landmarks import draw_landmarks, draw_parts
# from get_lengths import *
# from identify_faces import *
# from calc_weights import calc_weights

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
    #TODO: Ev: add a pretty CLI for users
    image_path = path_to_all_dataset + "ev.JPG"
    # image_path = path_to_all_dataset + "14.jpg"
    draw_landmarks(image_path, circle_type = "dominant")
    draw_parts(image_path)

