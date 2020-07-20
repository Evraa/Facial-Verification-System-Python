def rename_datset():
    import os
    path = '../dataset/'
    files = os.listdir(path)
    i = 0
    for img_file in files:
        file_path = path + img_file
        new_name = path + str(i) + '.jpg'
        i+=1
        os.rename(file_path,new_name)


