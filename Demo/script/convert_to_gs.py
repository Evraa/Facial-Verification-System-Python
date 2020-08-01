import os
from PIL import Image 


#WILL NEED THEM
# from glob import glob
# human_files = np.array(glob("/data/lfw/*/*"))



# dataset_path = '../dataset/yalefaces/'
# folder_names = os.listdir(dataset_path)

# for folder_name in folder_names:
#     folder_path = dataset_path+folder_name+'/'
#     new_name = dataset_path+folder_name[:len(folder_name)-2] + '/'
#     print (new_name)
#     os.rename(folder_path,new_name)
# for folder_name in folder_names:
#     folder_path = dataset_path+folder_name+'/'
#     new_folder_path = dataset_path+folder_name+'_g/'
#     os.mkdir(new_folder_path)
#     image_names = os.listdir(folder_path)
#     for image_name in image_names:
#         image_path = folder_path+image_name
#         new_image_path = new_folder_path+image_name
#         image_file = Image.open(image_path)
#         image_file = image_file.convert('L')
#         image_file.save(new_image_path)
