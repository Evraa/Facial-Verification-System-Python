import statistics

import pandas

from auxilary import *

data = read_csv(fileName=path_to_csv_key_points)
base = data["base_point"]
features = data[data.columns[3:15]]


# for each image set
def get_set_devs(j):
    df = {  # 6 major points
        'left_ebr1': [],
        'left_ebr2': [],
        'right_ebr1': [],
        'right_ebr2': [],
        'left_eye1': [],
        'left_eye2': [],
        'right_eye1': [],
        'right_eye2': [],
        'nose1': [],
        'nose2': [],
        'mouth1': [],
        'mouth2': []
    }
    dists = {'left_ebr': [],
             'right_ebr': [],
             'left_eye': [],
             'right_eye': [],
             'nose': [],
             'mouth': []}
    df = pandas.DataFrame(df)
    num = 0
    for i in data.index:
        # get the indices of values
        if data.loc[i, "image_set"] == j:
            num = num + 1
            temp = []
            for f in features:
                a = [int(i) for i in [x.strip('[]') for x in base[i].split(',')]]
                b = [int(i) for i in [x.strip('[]') for x in features.loc[i, f].split(',')]]
                temp.append(distance_two_points(a, b))
            df.loc[num] = temp
    dists['left_ebr'] = df.iloc[:, 0:2].mean(axis=1)
    dists['right_ebr'] = df.iloc[:, 2:4].mean(axis=1)
    dists['left_eye'] = df.iloc[:, 4:6].mean(axis=1)
    dists['right_eye'] = df.iloc[:, 6:8].mean(axis=1)
    dists['nose'] = df.iloc[:, 8:10].mean(axis=1)
    dists['mouth'] = df.iloc[:, 10:12].mean(axis=1)
    dists = pandas.DataFrame(dists)
    stds = dists.std().tolist()
    return stds


def get_weight():
    compareFaces()


def calc_weights():
    '''
            TODO:
                + Find differences between all the combinations of each person's images
                    (person 1 has image 1,2,3) -> diff_1_2, diff_1_3, diff_2_3
                    if 4 images -> diff_1_2, diff_1_3, diff_1_4, diff_2_3, diff_2_4, diff_3_4

                + store each set of images in a row in the result dataframe

                + check on the range of values (manually, or using some code)

                + convert the values into the largest range (not manually of course)

                + add these values

                + normalize them -> (value[0] = value[0] / sum of all)

                + weights = 1 - normalized_version (remember the lower the better)

                + weights *= 100

                + done :D
        '''
    weight = {"features": ['left_ebr', 'right_ebr', 'left_eye', 'right_eye', 'nose', 'mouth']}
    for sets in np.unique(data["image_set"].tolist()):
        deviations = weigh_data(get_set_devs(sets))
        w = []
        for i in deviations:
            factor = 1 / len(deviations)
            w.append((factor - i) + factor)
        weight[sets] = w
    weight = pandas.DataFrame(weight).set_index('features').T
    return weight


# takes a list of standard deviations
def weigh_data(stddev):
    s = sum(stddev)
    l = []
    for i in range(len(stddev)):
        l.append(stddev[i] / s)
    return l


def store_csv(dataframe, path):
    '''
        this function is called implicity to store the new dictionary rows added.
    '''
    dataframe.to_csv(path, index=False)


store_csv(calc_weights(), path_to_weighted_set_data)
