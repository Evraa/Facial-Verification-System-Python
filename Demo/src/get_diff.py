import itertools
from random import random, sample, choices, randint

import os, math

import pandas

import auxilary
import copy
import numpy as np


def comnimationals(n):
    f = math.factorial
    return int(f(n) / f(2) / f(n - 2))


def get_points(string):
    print(string)
    string = string.replace('[', '')
    string = string.replace(']', '')
    string = string.split(',')
    points = [int(string[0]), int(string[1])]
    return points


def mse_diff(row1, row2):
    diff_1 = pandas.to_numeric(row1)
    diff_2 = pandas.to_numeric(row2)

    return np.subtract(diff_1, diff_2).to_list()


def clac_diff(row):
    diff_1 = []
    rowC = copy.deepcopy(row)
    rowC = rowC[2:]
    base = get_points(rowC[0])

    rowC = rowC[1:]
    for r in rowC:
        r_p = get_points(r)
        diff_1.append(auxilary.distance_two_points(base, r_p))
    return diff_1


#formerly get_diff
def get_binary_classification(key_points_path, image_dir):
    main_data = auxilary.read_csv(key_points_path)
    # create diff csv
    columns = ['inputs', 'label']
    path = '../csv_files/svm_set2.csv'

    if not os.path.exists(path):
        auxilary.create_csv(path, columns)

    dataframe = auxilary.read_csv(fileName=path)
    print("a", len(main_data))

    print("b", len(dataframe[dataframe.label == 0]))
    num_sets = int(main_data[['output']].nunique())
    list_of_people = main_data['output'].unique()
    data_size = 1000

    # # GET LABEL 1
    def get_ones():
        for person in list_of_people:
            print(person)
            dataframe = auxilary.read_csv(fileName=path)
            if len(dataframe.index) > data_size:
                break

            df_values = main_data.iloc[:, 0:22]
            df_filter = df_values[main_data['output'] == person]
            values_per_set = math.ceil(data_size / num_sets)
            pairs = list(itertools.combinations(df_filter.index, 2))
            if len(pairs) > values_per_set:
                pairs = pairs[0:int(values_per_set)]

            for pair in pairs:
                row1 = df_values.iloc[pair[0]]
                row2 = df_values.iloc[pair[1]]
                diff = mse_diff(row1,row2)
                row_dict = {
                    'inputs': [],
                    'label': 1
                }
                row_dict['inputs'].append(diff)
                dataframe = auxilary.read_csv(fileName=path)
                auxilary.add_row(dataframe, row_dict, fileName=path)

    # RUN ONCE ONLY
    # get_ones()

    # GET LABEL 0
    curr_size = len(dataframe.index)

    def get_zeros(list_of_people, size, index):
        while size > 0:
            index2 = len(list_of_people) - ( index + 1)
            print(index,index2)
            if index2 <= index:
                list_of_people = list_of_people[1:]
                size = size - 1
                get_zeros(list_of_people, size, 0)
                break
            df = main_data.iloc[:, 0:22]
            row1 = df[main_data['output'] == list_of_people[index]].sample().squeeze()
            row2 = df[main_data['output'] == list_of_people[index2]].sample().squeeze()
            diff = mse_diff(row1, row2)
            row_dict = {
                'inputs': [],
                'label': 0
            }
            row_dict['inputs'].append(diff)
            df = auxilary.read_csv(fileName=path)
            print(row_dict['inputs'])
            auxilary.add_row(df, row_dict, fileName=path)
            size = size - 1
            index = index + 1
        return

    #RUN ONCE ONLY
    get_zeros(list_of_people, curr_size, 0)

    dataframe = auxilary.read_csv(fileName=path)
    print ("c",len(dataframe))
    print ("d",len(dataframe[dataframe.label == 1] ) )
    print ("d",len(dataframe[dataframe.label == 0] ) )

# get_multiclass('../csv_files/embedded_2.csv')
get_binary_classification('../csv_files/embedded_2.csv', '/Demo/dataset/main_data')
