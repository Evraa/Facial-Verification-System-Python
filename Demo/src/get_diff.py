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

def mse_diff (diff_1,diff_2):
    diff_1 = np.array(diff_1) 
    diff_2 = np.array(diff_2) 
    
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

def get_diff(key_points_path):

    main_data = auxilary.read_csv(key_points_path)
    #create diff csv
    columns = ['inputs','label']
    path = '../csv_files/csv_differences.csv'

    if not os.path.exists(path):
        auxilary.create_csv(path, columns)
    
    # dataframe = auxilary.read_csv(fileName=path)
    # print (len(dataframe))

    # print (len(dataframe[dataframe.label == 0] ) )
    # values = auxilary.strings_to_lists (dataframe['inputs'][0])
    num_sets = 15

    #append similars
    # for i in range (num_sets):
    #     set_i_mask = main_data['image_set'] == (i+1)
    #     set_i = main_data[set_i_mask]
    #     # iters = comnimationals(len(set_i))
    #     # for it in range (iters):
    #     for index, row in set_i.iterrows():
    #         diff_1 = clac_diff(row)
            
    #         for index_2,row_2 in set_i.iterrows():
    #             if index >= index_2:
    #                 continue
    #             diff_2 = clac_diff(row_2)
    #             diff = mse_diff (diff_1,diff_2)
    #             row_dict = {
    #                 'inputs': [],
    #                 'label':1
    #             }
    #             row_dict['inputs'].append(diff)
    #             dataframe = auxilary.read_csv(fileName=path)
    #             auxilary.add_row(dataframe, row_dict, fileName=path)

    #append differents
    uniques = []
    for i in range (15):
        # set_i_mask = main_data['image_set'] == (i+1)
        set_i = main_data.loc[ main_data['image_set']== (i+1) ].iloc[6]
        uniques.append(set_i)

    for i,unique in enumerate(uniques):
        diff_1 = clac_diff(unique)

        for j,unique_2 in enumerate(uniques):
            if i >= j:
                continue
            diff_2 = clac_diff(unique_2)
            diff = mse_diff (diff_1,diff_2)
            row_dict = {
                'inputs': [],
                'label':0
            }
            row_dict['inputs'].append(diff)
            dataframe = auxilary.read_csv(fileName=path)
            auxilary.add_row(dataframe, row_dict, fileName=path)
            
    dataframe = auxilary.read_csv(fileName=path)
    print (len(dataframe))

    print (len(dataframe[dataframe.label == 0] ) )