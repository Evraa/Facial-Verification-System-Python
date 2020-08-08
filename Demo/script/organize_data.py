import os
import numpy as np
from glob import glob
import shutil

main_dir = '../dataset/lfw/*'
sub_dir = '../dataset/main_data/'
folders = np.array(glob(main_dir))

new_c = 0
for folder in folders:
    files = os.listdir(folder)
    count = len(files)
    
    if count >= 7:
        #move it to heaven
        folder_name = folder.split('/')[2]
        folder_name = folder.split('\\')[1] 
        dst = sub_dir+folder_name
        src = folder
        shutil.copytree(src,dst)
        new_c += 1
print (new_c)

# count_dec = sorted(count,reverse=True)
# print (count_dec)