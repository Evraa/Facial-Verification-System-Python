'''
    Script to print top K folders with largest no. of images
'''

import numpy as np
from glob import glob
import os

def go ():
    main_path = "../dataset/main_data/"
    folders = os.listdir(main_path)
    dic = {}
    values=[]
    for f in folders:
        dic[f]  = len(os.listdir(main_path+f))
        values.append(len(os.listdir(main_path+f)))
    # print({key: value for key, value in sorted(dic.items(), key=lambda item: item[1])})
    print (sorted(set(values)))

go()