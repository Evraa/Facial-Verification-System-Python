import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage.transform import rotate
from skimage.util import random_noise
import os

def flip_hor(image):
    return np.fliplr(image).astype('uint8')

def rotate_right(image):
    return rotate(image, angle=-45).astype('uint8')

def rotate_left(image):
    return rotate(image, angle=45).astype('uint8')

def random_noise_fn (image):
    return random_noise(image).astype('uint8')

def gaus_blur(image):
    return cv2.GaussianBlur(image, (5, 5),0)  .astype('uint8')

def shift_right(image):
    dst = np.zeros_like(image)
    col = image.shape[1]
    dst[:, 40:, :] = image[:, :col-40, :]
    dst[:, :40, :] = image[:, col-40:, :]
    return dst.astype('uint8')

def shift_left(image):
    dst = np.zeros_like(image)
    col = image.shape[1]
    dst[:, :col-40, :] = image[:, 40:, :]
    dst[:, col-40:, :] = image[:, :40, :]
    return dst.astype('uint8')

def subplot_two_images(image1, image2,image3, image4,image5, image6,image7, image8 ):
    plt.clf()
    plt.subplot(3,3,1)
    plt.title('Original')
    plt.imshow(image1)
    plt.subplot(3,3,2)
    plt.title('Horizontal flip')
    plt.imshow(image2)
    plt.subplot(3,3,3)
    plt.title('Rotate Right')
    plt.imshow(image3)
    plt.subplot(3,3,4)
    plt.title('Rotate Left')
    plt.imshow(image4)
    plt.subplot(3,3,5)
    plt.title('Random Noise')
    plt.imshow(image5)
    plt.subplot(3,3,6)
    plt.title('Blurring')
    plt.imshow(image6)
    plt.subplot(3,3,7)
    plt.title('Shift Right')
    plt.imshow(image7)
    plt.subplot(3,3,8)
    plt.title('Shift Left')
    plt.imshow(image8)
    plt.show()
 
def store_17(image, store_path, image_name):
    name = image_name.split('.')[0]
    store_path = store_path +'/'
    
    sh_l = shift_left(image)
    io.imsave(fname=store_path+name+"_shift_L.jpg", arr=sh_l)
    
    gb = gaus_blur(sh_l)
    io.imsave(fname=store_path+name+"_shift_L_G.jpg", arr=gb)

    rn = random_noise_fn(sh_l)
    io.imsave(fname=store_path+name+"_shift_L_RN.jpg", arr=rn)

    sh_r = shift_right(image)
    io.imsave(fname=store_path+name+"_shift_R.jpg", arr=sh_r)

    gb = gaus_blur(sh_r)
    io.imsave(fname=store_path+name+"_shift_R_G.jpg", arr=gb)

    rn = random_noise_fn(sh_r)
    io.imsave(fname=store_path+name+"_shift_R_RN.jpg", arr=rn)

    r_l = rotate_left(image)
    io.imsave(fname=store_path+name+"_rotate_L.jpg", arr=r_l)

    gb = gaus_blur(r_l)
    io.imsave(fname=store_path+name+"_rotate_L_G.jpg", arr=gb)

    rn = random_noise_fn(r_l)
    io.imsave(fname=store_path+name+"_rotate_L_RN.jpg", arr=rn)

    r_r = rotate_right(image)
    io.imsave(fname=store_path+name+"_rotate_R.jpg", arr=r_r)

    gb = gaus_blur(r_r)
    io.imsave(fname=store_path+name+"_rotate_R_G.jpg", arr=gb)

    rn = random_noise_fn(r_r)
    io.imsave(fname=store_path+name+"_rotate_R_RN.jpg", arr=rn)

    h_f = flip_hor(image)
    io.imsave(fname=store_path+name+"flip_H.jpg", arr=h_f)

    gb = gaus_blur(h_f)
    io.imsave(fname=store_path+name+"flip_H_G.jpg", arr=gb)

    rn = random_noise_fn(h_f)
    io.imsave(fname=store_path+name+"flip_H_RN.jpg", arr=rn)

    gs = gaus_blur(image)
    io.imsave(fname=store_path+name+"gauss.jpg", arr=gs)

    rn = random_noise_fn(image)
    io.imsave(fname=store_path+name+"random_noise.jpg", arr=rn)

def store_12(image, store_path, image_name):
    name = image_name.split('.')[0]
    store_path = store_path +'/'
    
    sh_l = shift_left(image)
    io.imsave(fname=store_path+name+"_shift_L.jpg", arr=sh_l)
    
    rn = random_noise_fn(sh_l)
    io.imsave(fname=store_path+name+"_shift_L_RN.jpg", arr=rn)

    sh_r = shift_right(image)
    io.imsave(fname=store_path+name+"_shift_R.jpg", arr=sh_r)

    
    rn = random_noise_fn(sh_r)
    io.imsave(fname=store_path+name+"_shift_R_RN.jpg", arr=rn)

    r_l = rotate_left(image)
    io.imsave(fname=store_path+name+"_rotate_L.jpg", arr=r_l)

    rn = random_noise_fn(r_l)
    io.imsave(fname=store_path+name+"_rotate_L_RN.jpg", arr=rn)

    r_r = rotate_right(image)
    io.imsave(fname=store_path+name+"_rotate_R.jpg", arr=r_r)

    rn = random_noise_fn(r_r)
    io.imsave(fname=store_path+name+"_rotate_R_RN.jpg", arr=rn)

    h_f = flip_hor(image)
    io.imsave(fname=store_path+name+"flip_H.jpg", arr=h_f)

    rn = random_noise_fn(h_f)
    io.imsave(fname=store_path+name+"flip_H_RN.jpg", arr=rn)

    gs = gaus_blur(image)
    io.imsave(fname=store_path+name+"gauss.jpg", arr=gs)

    rn = random_noise_fn(image)
    io.imsave(fname=store_path+name+"random_noise.jpg", arr=rn)

def store_7(image, store_path, image_name):
    name = image_name.split('.')[0]
    store_path = store_path +'/'
    
    sh_l = shift_left(image)
    io.imsave(fname=store_path+name+"_shift_L.jpg", arr=sh_l)
    
    sh_r = shift_right(image)
    io.imsave(fname=store_path+name+"_shift_R.jpg", arr=sh_r)

    
    r_l = rotate_left(image)
    io.imsave(fname=store_path+name+"_rotate_L.jpg", arr=r_l)

    
    r_r = rotate_right(image)
    io.imsave(fname=store_path+name+"_rotate_R.jpg", arr=r_r)

    
    h_f = flip_hor(image)
    io.imsave(fname=store_path+name+"flip_H.jpg", arr=h_f)

    
    gs = gaus_blur(image)
    io.imsave(fname=store_path+name+"gauss.jpg", arr=gs)

    rn = random_noise_fn(image)
    io.imsave(fname=store_path+name+"random_noise.jpg", arr=rn)

def store_2(image, store_path, image_name):
    name = image_name.split('.')[0]
    store_path = store_path +'/'
    
    gs = gaus_blur(image)
    io.imsave(fname=store_path+name+"gauss.jpg", arr=gs)

    rn = random_noise_fn(image)
    io.imsave(fname=store_path+name+"random_noise.jpg", arr=rn)

if __name__ == "__main__":
    dataset = '../dataset/main_data/*'
    folders = np.array(glob(dataset))

    print ("Refer to the author for the correct sequence of execution!")
    # for image_path in folders:
    #     source = image_path
    #     dst = image_path.replace("main_data", "augmented")
    #     os.rename(source, dst)
    
    # rand_int = np.random.randint(0, len(folders)-1)
    # img_path = folders[rand_int]
    # image = io.imread(img_path)
    # flipped = flip_hor(image)
    # print (image[0])
    # input('e')
    
    # for e, folder in enumerate(folders):
    #     print (f'Folder: {e}/{len(folders)}')
    #     num_imgs = len(os.listdir(folder))
    #     folder_path = '../dataset/augmented/'+(folder.split('/')[2].split('\\')[1])
    #     os.mkdir(folder_path)
    #     if num_imgs>=7 and num_imgs<20:
    #         #17 features
    #         image_paths = os.listdir(folder)
    #         for image_path in image_paths:
    #             img_path = folder
    #             path = img_path+'/'+image_path
    #             image = io.imread(path)
    #             store_17(image, store_path = folder_path, image_name=image_path)

    #     elif num_imgs>=20 and num_imgs<40:
    #         #12
    #         image_paths = os.listdir(folder)
    #         for image_path in image_paths:
    #             img_path = folder
    #             path = img_path+'/'+image_path
    #             image = io.imread(path)
    #             store_12(image, img_path, image_name=image_path)

    #     elif num_imgs>=40 and num_imgs<75:
    #         image_paths = os.listdir(folder)
    #         for image_path in image_paths:
    #             img_path = folder
    #             path = img_path+'/'+image_path
    #             image = io.imread(path)
    #             store_12(image, img_path, image_name=image_path)

    #     elif num_imgs>=100 and num_imgs<200:
    #         image_paths = os.listdir(folder)
    #         for image_path in image_paths:
    #             img_path = folder
    #             path = img_path+'/'+image_path
    #             image = io.imread(path)
    #             store_7(image, img_path, image_name=image_path)

    #     elif num_imgs>=200 and num_imgs<500:
    #         image_paths = os.listdir(folder)
    #         for image_path in image_paths:
    #             img_path = folder
    #             path = img_path+'/'+image_path
    #             image = io.imread(path)
    #             store_2(image, img_path, image_name=image_path)

    # rand_int = np.random.randint(0, len(human_files)-1)
    # img_path = human_files[rand_int]
    # image = io.imread(img_path)
    # subplot_two_images(image, flip_hor(image),rotate_right(image),
    #                             rotate_left(image),random_noise_fn (image), gaus_blur(image),
    #                             shift_right(image), shift_left(image) )

    for e, folder in enumerate(folders):
        num_imgs = len(os.listdir(folder))
        
        if num_imgs>700 :
            image_paths = os.listdir(folder)
            for image_path in image_paths:
                rand_int = np.random.randint(0, 2)
                if rand_int == 0:
                    deleted_file_path = folder + '/' + image_path
                    os.remove(deleted_file_path)
            print (f"this was: {num_imgs} and became {len(os.listdir(folder))}")
        
        elif num_imgs>500:
            image_paths = os.listdir(folder)
            for image_path in image_paths:
                rand_int = np.random.randint(0, 3)
                if rand_int == 0:
                    deleted_file_path = folder + '/' + image_path
                    os.remove(deleted_file_path)
            print (f"this was: {num_imgs} and became {len(os.listdir(folder))}")