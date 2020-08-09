import itertools
from random import random, sample, choices, randint

import os, math
import auxilary
import copy
import numpy as np
def comnimationals(n):
    f = math.factorial
    return int(f(n) / f(2) / f(n-2))


def get_points(string):
    string = string.replace('[', '') 
    string = string.replace(']', '') 
    string = string.split(',')
    points = [int(string[0]),int(string[1])]
    return points

def mse_diff (row1,row2):
    diff_1 = np.array(clac_diff(row1))
    diff_2 = np.array(clac_diff(row2))

    return abs(np.subtract(diff_1,diff_2))

def clac_diff(row):
    diff_1 = []
    rowC = copy.deepcopy(row)
    rowC = rowC[2:]
    base = get_points(rowC[0])

    rowC = rowC[1:]
    for r in rowC:
        r_p = get_points(r)
        diff_1.append(auxilary.distance_two_points(base,r_p))
    return diff_1

def get_diff(key_points_path, image_dir):

    main_data = auxilary.read_csv(key_points_path)
    #create diff csv
    columns = ['inputs','label']
    path = '../csv_files/csv_differences_prof.csv'

    if not os.path.exists(path):
        auxilary.create_csv(path, columns)
    
    dataframe = auxilary.read_csv(fileName=path)
    print ("a",len(main_data))


    print ("b",len(dataframe[dataframe.label == 0] ) )
    # values = auxilary.strings_to_lists (dataframe['inputs'][0])
    num_sets = main_data['image_set'].nunique()


    # GET LABEL 1
    set_i_mask = main_data['image_set'] == "Vladimir_Putin"
    set_i = main_data[set_i_mask]
    # for index, row in set_i.iterrows():
    #     for index2, row2 in set_i.iterrows():
    #         if index >= index2:
    #             continue
    #         diff = mse_diff(row, row2)
    #         row_dict = {
    #             'inputs': [],
    #             'label': 1
    #         }
    #         row_dict['inputs'].append(diff)
    #         dataframe = auxilary.read_csv(fileName=path)
    #         auxilary.add_row(dataframe, row_dict, fileName=path)
    #         print(index,index2)

    # GET LABEL 0
    num_0 = len(dataframe)
    photos_per_set = len(dataframe)/num_sets
    if photos_per_set > 7:
        photos_per_set = 7
    sets = auxilary.mylistdir(image_dir)
    set_indx = 0
    r = choices(sets, k=num_0)
    for cset in r:
        dataframe = auxilary.read_csv(fileName=path)
        compare_set = main_data[main_data['image_set'] == cset]
        set_indx = set_indx + 1
        rand_num = randint(0, len(compare_set) - 1)
        compare_row = compare_set.iloc[rand_num]
        row = set_i.iloc[randint(0, len(set_i)-1)]
        diff = mse_diff(row,compare_row)
        row_dict = {
            'inputs': [],
            'label': 0
        }
        row_dict['inputs'].append(diff)
        auxilary.add_row(dataframe, row_dict, fileName=path)


    # for row in set_i.iterrows():
    #     dataframe = auxilary.read_csv(fileName=path)
    #     compare_set = main_data[main_data['image_set'] == r[set_indx]]
    #     set_indx = set_indx + 1
    #     rand_num = randint(0, len(compare_set) - 1)
    #     compare_row = compare_set.iloc[rand_num]
    #     diff = mse_diff(row[1], compare_row)
    #     # print("DIFFERENCE: ", diff)
    #     row_dict = {
    #         'inputs': [],
    #         'label': 0
    #     }
    #     row_dict['inputs'].append(diff)
    #     auxilary.add_row(dataframe, row_dict, fileName=path)

    #append similars
    # for i in range (num_sets):
        # set_i_mask = main_data['image_set'] == (i+1)
        # set_i = main_data[set_i_mask]
        # # iters = comnimationals(len(set_i))
        # # for it in range (iters):
        # for index, row in set_i.iterrows():
        #     diff_1 = clac_diff(row)
        #
        #     for index_2,row_2 in set_i.iterrows():
        #         if index >= index_2:
        #             continue
        #         diff_2 = clac_diff(row_2)
        #         diff = mse_diff (diff_1,diff_2)
        #         row_dict = {
        #             'inputs': [],
        #             'label':1
        #         }
        #         row_dict['inputs'].append(diff)
        #         dataframe = auxilary.read_csv(fileName=path)
        #         auxilary.add_row(dataframe, row_dict, fileName=path)
    #
    # #append differents
    # uniques = []
    # for i in range (15):
    #     # set_i_mask = main_data['image_set'] == (i+1)
    #     set_i = main_data.loc[ main_data['image_set'] == (i+1) ].iloc[6]
    #     uniques.append(set_i)
    #
    # for i,unique in enumerate(uniques):
    #     diff_1 = clac_diff(unique)
    #
    #     for j,unique_2 in enumerate(uniques):
    #         if i >= j:
    #             continue
    #         diff_2 = clac_diff(unique_2)
    #         diff = mse_diff (diff_1,diff_2)
    #         row_dict = {
    #             'inputs': [],
    #             'label':0
    #         }
    #         row_dict['inputs'].append(diff)
    #         dataframe = auxilary.read_csv(fileName=path)
    #         auxilary.add_row(dataframe, row_dict, fileName=path)
            
    # dataframe = auxilary.read_csv(fileName=path)
    print ("c",len(dataframe))
    print ("d",len(dataframe[dataframe.label == 1] ) )
    print ("d",len(dataframe[dataframe.label == 0] ) )