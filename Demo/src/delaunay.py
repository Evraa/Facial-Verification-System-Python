import scipy
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from facial_landmarks import predict_shapes,np,cv2
from auxilary import dominant_key_points

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
        

    return img


    # print (points[:,0])
    # print (points[:,1])
    # print(tri.simplices.copy())
    # plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
    # plt.plot(points[:,0], points[:,1], 'o')
    # plt.show()

    return 

def draw_line (x_1,y_1,x_2,y_2,img):
    cv2.line(img, (x_1, y_1), (x_2, y_2), (168, 185, 90), 2) 
    return


def get_delaunay_points(image_path):
    shape,_,image = predict_shapes(image_path)
    shape_d = shape[dominant_key_points]
    imgage = plotter(shape_d,image)
    cv2.imshow('Delaunay', imgage) 
    cv2.waitKey(0)
