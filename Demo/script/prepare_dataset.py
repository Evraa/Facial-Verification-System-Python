import os

path = "../dataset/yalefaces/"
# for i in range (15):
#     folder_name = path + str(i+1)
#     if not os.path.exists(folder_name):
#         os.mkdir(folder_name)

yalefaces = os.listdir(path)
for folder_name in yalefaces:
    folder_path = path +folder_name
    images = os.listdir(folder_path)
    for i,image_name in enumerate(images):
        image_path_old = folder_path +'/'+ image_name
        name = image_name.split('.')[0]
        image_path_new = folder_path +'/'+ name +'_'+str(i)+ ".gif"
        os.rename(image_path_old,image_path_new)