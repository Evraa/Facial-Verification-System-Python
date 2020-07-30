import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import imread,subplot,imshow,show
from skimage import color
from skimage import io
from PIL import Image 



def show_images(images_list, name_list):
    '''
        Simply takes an array of images and show them all together labeld
    '''
    pass


def random_numbers(L,H):
    return np.random.random_integers(L,H)

def get_random_image_path(dataset_path):
    folder_numb = random_numbers(1,15)
    file_numb = random_numbers(0,10)
    folder_path = dataset_path + str(folder_numb) + '/'
    images = os.listdir(folder_path)
    image_name = images[file_numb]
    image_path = folder_path + image_name
    if not os.path.exists(image_path):
        print (f'ERROR: That image path {image_path} DOES NOT exist')
        return False
    return image_path


def show_tests(dataset_path):
    '''
        + Takes the folder of sets of images
        + Randomly choose an image, and show [Identicalls, simillars, Fasle call]
    '''
    image_path = get_random_image_path(dataset_path)
    # img=mpimg.imread(image_path)
    # img = io.imread(image_path, as_gray=True)
    image_file = Image.open('testimage.gif')
    # image_file = image_file.convert('L')
    # image_file.save('testimage.gif')
    imgplot = plt.imshow(image_file)
    plt.show()
    
    # img = color.rgb2gray(io.imread(image_path))
    # # image = imread(image_path)  
    # print ((img))
    # imgplot = plt.imshow(img)
    # plt.show()
  