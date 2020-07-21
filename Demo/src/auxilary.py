import pandas as pd
import os
import numpy as np


def create_demo(fileName='csv_example.csv'):
    '''
        Creates a csv file with the 7 points and the firs row includes the header
    '''
    my_dict = {'image_name': [],
               # 7 major points
               'Eye_br_L': [],
               'Eye_br_R': [],
               'Eye_soc_L': [],
               'Eye_soc_R': [],
               'Nostril_L': [],
               'Nostril_R': [],
               'Moustache': [],
               'Face_Width': [],
               'Face_Height': []
               }

    # convert it into dataframe
    df = pd.DataFrame(my_dict)
    # transfer into csv file
    df.to_csv(fileName, index=False)


def read_csv(fileName='csv_example.csv'):
    '''
        returns the csv file we're working on
    '''
    df = pd.read_csv('csv_example.csv')
    return df


def store_csv(dataframe, fileName='csv_example.csv'):
    '''
        this function is called implicity to store the new dictionary rows added.
    '''
    dataframe.to_csv(fileName, index=False)


def add_row(dataframe, row_dict, fileName='csv_example.csv'):
    '''
        Append row of data, and store it.
    '''
    if len(row_dict) < 7:
        print("error: row length is incorrect!", row_dict)
        return None

    row = pd.DataFrame(row_dict)
    dataframe_concatenatd = pd.concat([dataframe, row], ignore_index=True)
    store_csv(dataframe=dataframe_concatenatd, fileName=fileName)
    return dataframe_concatenatd


def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
	'''
		Convert the `shape` returned from dlib into ndarray
	'''
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords


# compares all faces in the database to one face
# returns two lists: similar to and same as
def compareFaces(indx, rw, d, features, x_scale,threshold_isSame, threshold_isSimilar):
    similar = []
    same = []
    # each face
    for i, r in d.iterrows():
        averagediff = []
        if not i == indx:
            # each feature
            for f in range(0, 6):
                xscale = rw[features[f]] / rw[x_scale]
                xscale2 = r[features[f]] / r[x_scale]
                # n = compareRatios(xscale, xscale2)
                n = weightedCompareRatios(xscale, xscale2, features[f])
                averagediff.append(n)
            # x = (sum(averagediff) / len(averagediff)) * 1000
            x = (sum(averagediff)) * 1000
            print("Score for", d['image_name'][i], "=", x)
            r = insertFaces(x,threshold_isSame, threshold_isSimilar)
            if r == 2:
                same.append(d['image_name'][i])
            elif r == 1:
                similar.append(d['image_name'][i])
    pair = [similar, same]
    return pair


# returns simple difference between ratios
def compareRatios(int1, int2):
    v = abs(int1 - int2)
    return v

# CHANGE WEIGHTING OF FEATURES
def weights(x):
    return {
        'Eye_br_L': 0.166666667,
        'Eye_br_R': 0.166666667,
        'Eye_soc_L': 0.166666667,
        'Eye_soc_R': 0.166666667,
        'Nostril_L': 0.166666667,
        'Nostril_R': 0.166666667
    }.get(x)  # 5 is default if x not found

# returns simple difference between ratios
def weightedCompareRatios(int1, int2, feature):
    weight = weights(feature)
    v = abs(int1 - int2) * weight
    return v


# inserts the picture name into the new columns depending of threshold
def insertFaces(v,threshold_isSame, threshold_isSimilar):
    if v < threshold_isSimilar:
        if v < threshold_isSame:
            return 2
        else:
            return 1
    else:
        return 0
