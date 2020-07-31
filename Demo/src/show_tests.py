import numpy as np
import os,math
import matplotlib.pyplot as plt
from PIL import Image 
import matplotlib.image as mpimg
import facial_landmarks
import auxilary

def show_image(images_list,image_names):
    '''
        Simply takes an array of images and show them all together labeld
    '''
    images_list_read = []
    for image_path in images_list:
        img = mpimg.imread(image_path)
        images_list_read.append(img)
    list_length = len(images_list)
    size = math.ceil(math.sqrt(list_length))
    _, axs = plt.subplots(size, size)
    axs = axs.flatten()    
    for img, image_name,ax in zip(images_list_read, image_names,axs):
        ax.imshow(img)
        ax.set_title(image_name)
        ax.axis('off')

    for i in range (len(images_list),len(axs)):
        axs[i].axis('off')
    plt.show()

    

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
    return image_path, image_name

def show_one_image(orig_image_path,orig_image_name):
    img = mpimg.imread(orig_image_path)
    imgplot = plt.imshow(img)
    plt.show()

def mse_diff (diff_1,diff_2):
    diff_1 = np.array(diff_1) 
    diff_2 = np.array(diff_2) 
    
    return abs(np.subtract(diff_1,diff_2))

def show_tests(dataset_path,clf,detector,predictor):
    '''
        + Takes the folder of sets of images
        + Randomly choose an image, and show [Identicalls, simillars, Fasle call]

        Now that we have an image, let's compare it to every other image in the dataset
        1- get facials of the mian image...then lengths
        2- for each image in the data set other than the original
            1- get facials
            2- calc distances

            3- clac differences with the original
            4- show it to the clf

            5- state the prediction
    '''
    # orig_image_path, orig_image_name = get_random_image_path(dataset_path)
    orig_image_path = '../dataset/yalefaces/6/subject06_0.gif'
    orig_image_name = "subject01_0.gif"
    orig_key_points,state = facial_landmarks.get_key_points(orig_image_path,detector,predictor)
    while state is False:
        orig_image_path, orig_image_name = get_random_image_path(dataset_path)
        orig_key_points,state = facial_landmarks.get_key_points(orig_image_path,detector,predictor)
    orig_base_point = orig_key_points[auxilary.fixed_key_point]
    orig_key_points = orig_key_points[auxilary.dominant_key_points]
    orig_lengths = auxilary.calc_lengths(orig_key_points,orig_base_point)
    
    folder_names = os.listdir(dataset_path)
    # folder_names = ['4']
    identicalls = []
    identicalls_names = []
    similars = []
    similars_names = []
    
    show_one_image(orig_image_path,orig_image_name)
    for folder_name in folder_names:
        folder_path = dataset_path + folder_name + '/'
        image_names = os.listdir(folder_path)
        for image_name in image_names:
            print (f'testing folder {folder_name} image {image_name}')
            image_path = folder_path + image_name
            if image_path == orig_image_path:
                continue
            key_points,state  = facial_landmarks.get_key_points(image_path,detector,predictor)
            if state is False:
                continue
            base_point  = key_points[auxilary.fixed_key_point]
            key_points  = key_points[auxilary.dominant_key_points]
            lengths     = auxilary.calc_lengths(key_points,base_point)

            #Calc differences
            diff_input = mse_diff (lengths,orig_lengths)
            #show it to clf
            diff_inputs = []
            diff_inputs.append(diff_input)
            prob = clf.predict_proba(diff_inputs)
            prob = prob[0][1]
            if prob <= 0.5:
                continue
            thsh= 0.85
            if prob <= thsh and prob >0.5:
                similars.append(image_path)
                similars_names.append(image_name)
                continue
            if prob > thsh:
                identicalls.append(image_path)
                identicalls_names.append(image_name)
                continue
    if len(similars) == 0 and len(identicalls) == 0:
        print ("No matchings at all")
        return
    if (len(identicalls)) == 0:
        print ("No identicalls")
        show_image(similars,similars_names)
        return
    if len(similars) == 0:
        print ("No identicalls")
        show_image(identicalls,identicalls_names)
        return

    show_image(identicalls,identicalls_names)
    show_image(similars,similars_names)
        