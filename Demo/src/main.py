from auxilary import path_to_all_dataset,os
from facial_landmarks import draw_landmarks, draw_parts
# from get_lengths import *
# from identify_faces import *
# from calc_weights import calc_weights
from delaunay import get_delaunay_points,store_shape_tri
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



    # TODO: Ev: add a pretty CLI for users

    files = os.listdir(path_to_all_dataset)
    for image_name in files:
        image_path = path_to_all_dataset + image_name
        store_shape_tri (image_path,image_name)

