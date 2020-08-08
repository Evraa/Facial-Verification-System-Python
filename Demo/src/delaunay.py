#GLOBAL IMPORTS
import scipy
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import os.path
import cv2

#LOCLA IMPORTS
# from facial_landmarks import predict_shapes
from auxilary import dominant_key_points,create_shape_tris,path_to_shape_tris,store_csv,read_csv


def plotter(points,img):
    tri = Delaunay(points)
    triangles = tri.simplices.copy()
    x_s = points[:,0]
    y_s = points[:,1]
    for triangle in triangles:
        p1 = triangle[0]
        p2 = triangle[1]
        p3 = triangle[2]

        x_1 = x_s[p1]
        x_2 = x_s[p2]
        x_3 = x_s[p3]
        
        y_1 = y_s[p1]
        y_2 = y_s[p2]
        y_3 = y_s[p3]

        draw_line(x_1,y_1, x_2,y_2,img)
        draw_line(x_1,y_1, x_3,y_3,img)
        draw_line(x_3,y_3, x_2,y_2,img)
        

    return img,triangles


def draw_line (x_1,y_1,x_2,y_2,img):
    cv2.line(img, (x_1, y_1), (x_2, y_2), (168, 185, 90), 2) 
    return


def get_delaunay_points(shape,image,returned = False):

    shape_d = shape[dominant_key_points]
    image,tris = plotter(shape_d,image)
    if returned:
        return shape,tris
    else:
        cv2.imshow('Delaunay Tri', image) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def store_shape_tri (image_path,image_name):
    shape,tris = get_delaunay_points(image_path,returned = True)
    if len(shape) == 0:
        return
    #create csv for tris and shape
    if not os.path.exists (path_to_shape_tris):
        create_shape_tris(fileName=path_to_shape_tris)
    #store values
    my_dict = {'image_name': None,
               # 7 major points
               'shape': None,
               'tris': None
               }
    my_dict['image_name'] = image_name
    my_dict['shape'] = shape
    my_dict['tris'] = tris
    
    dataframe = read_csv(fileName=path_to_shape_tris)
    n_rows = len(dataframe)
    dataframe.at[n_rows,'image_name'] = image_name
    dataframe.at[n_rows,'shape'] = shape
    dataframe.at[n_rows,'tris'] = tris
    store_csv(dataframe=dataframe, fileName=path_to_shape_tris)
    
    return