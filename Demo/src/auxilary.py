import pandas as pd
import os
import numpy as np

#GLOBALS allowed through out the whole project
path_to_csv_key_points = '../csv_files/csv_key_points.csv'
dominant_key_points = [17, 21, 22, 26, 36, 39, 42, 45, 32, 34, 48, 54]
fixed_key_point = 33
path_to_all_dataset = "../dataset/"
path_to_csv_lengths = '../csv_files/csv_lengths.csv'
path_to_shape_predictor = "../shape_predictor_68_face_landmarks.dat"
path_to_images_grouped = "../dataset/grouped/"

def create_demo(fileName=path_to_csv_lengths):
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

def create_key_points_data_frame(fileName=path_to_csv_key_points):
    '''
        Creates a csv file with the 7 points and the firs row includes the header
    '''
    my_dict = { 'image_set': [],
                'image_name': [],
                'base_point': [],
                'feat_17': [],
                'feat_21': [],
                'feat_22': [],
                'feat_26': [],
                'feat_36': [],
                'feat_39': [],
                'feat_42': [],
                'feat_45': [],
                'feat_32': [],
                'feat_34': [],
                'feat_48': [],
                'feat_54': []
    }

    # convert it into dataframe
    df = pd.DataFrame(my_dict)
    # transfer into csv file
    df.to_csv(fileName, index=False)


def read_csv(fileName=path_to_csv_key_points):
    '''
        returns the csv file we're working on
    '''
    df = pd.read_csv(fileName)
    return df


def store_csv(dataframe, fileName=path_to_csv_key_points):
    '''
        this function is called implicity to store the new dictionary rows added.
    '''
    dataframe.to_csv(fileName, index=False)


def add_row(dataframe, row_dict, fileName=path_to_csv_key_points):
    '''
        Append row of data, and store it.
    '''
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
        valid = 0
        if not i == indx:
            # each feature
            for f in range(7):
                xscale = rw[features[f]] / rw[x_scale]
                xscale2 = r[features[f]] / r[x_scale]
                # n = compareRatios(xscale, xscale2)
                # n = weightedCompareRatios(xscale, xscale2, features[f])
                if abs(xscale-xscale2) < 0.004:
                    valid += 1

                # averagediff.append(n)
            # x = (sum(averagediff) / len(averagediff)) * 1000
            # x = (sum(averagediff)) * 1000
            # print("Score for", d['image_name'][i], "=", x)
            
            r = insertFaces(valid,threshold_isSame, threshold_isSimilar)
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
# def insertFaces(v,threshold_isSame, threshold_isSimilar):
#     if v < threshold_isSimilar:
#         if v < threshold_isSame:
#             return 2
#         else:
#             return 1
#     else:
#         return 0

def insertFaces(v,threshold_isSame, threshold_isSimilar):
    if v >= threshold_isSimilar:
        if v >= threshold_isSame:
            return 2
        else:
            return 1
    else:
        return 0


def distance_two_points(x,y):
    x_diff = x[0] - y[0]
    x_pow = x_diff**2
    y_diff = x[1] - y[1]
    y_pow = y_diff**2
    return np.sqrt(x_pow + y_pow)

def calc_distances(image_name,shape):
    '''
        Calculate the distances:
            + Left eyebrow
            + Right eyebrow
            + Left eye socket
            + Right eye socket
            + Left nostril
            + Right nostril
            + Mouse
        
        then append them in the dictionary
    '''
    print (f'this is image: {image_name}')
    face_features = {   'image_name'    :[],
                        #7 major points
                        'Eye_br_L'      :[],
                        'Eye_br_R'      :[],
                        'Eye_soc_L'     :[],
                        'Eye_soc_R'     :[],
                        'Nostril_L'     :[],
                        'Nostril_R'     :[],
                        'Moustache'     :[],
                        'Face_Width'    :[],
                        'Face_Height'   :[]
                        }

    face_features['image_name'].append(image_name)
    # eyebrows
    face_features['Eye_br_L'].append(distance_two_points(shape[17],shape[21]))
    face_features['Eye_br_R'].append( distance_two_points(shape[22],shape[26]))
    #eye sockets
    face_features['Eye_soc_L'].append(distance_two_points(shape[36],shape[39]))
    face_features['Eye_soc_R'].append(distance_two_points(shape[42],shape[45]))
    #nostrils
    face_features['Nostril_L'].append(distance_two_points(shape[31],shape[32]))
    face_features['Nostril_R'].append(distance_two_points(shape[34],shape[35]))
    #Mouse
    face_features['Moustache'].append(distance_two_points(shape[48],shape[54]))
    #width
    face_features['Face_Width'].append(distance_two_points(shape[0],shape[16]))
    #height
    x = shape[19]
    y = shape[24]
    mid_point = ((x[0]+y[0])/2 ,(x[1]+y[1])/2 )
    face_features['Face_Height'] = distance_two_points(mid_point,shape[8])
    
    #store the data
    df = read_csv()
    add_row(df,face_features)


def store_keys(image_name, shape, set_number):
    my_dict = { 'image_set': [],
                'image_name': [],
                'base_point': [],
                'feat_17': [],
                'feat_21': [],
                'feat_22': [],
                'feat_26': [],
                'feat_36': [],
                'feat_39': [],
                'feat_42': [],
                'feat_45': [],
                'feat_32': [],
                'feat_34': [],
                'feat_48': [],
                'feat_54': []
    }
    my_dict['image_set'].append(set_number)
    my_dict['image_name'].append(image_name)
    my_dict['base_point'].append(fixed_key_point)
    my_dict['feat_17'].append(list(shape[17]))
    my_dict['feat_21'].append(list(shape[21]))
    my_dict['feat_22'].append(list(shape[22]))
    my_dict['feat_26'].append(list(shape[26]))
    my_dict['feat_36'].append(list(shape[36]))
    my_dict['feat_39'].append(list(shape[39]))
    my_dict['feat_42'].append(list(shape[42]))
    my_dict['feat_45'].append(list(shape[45]))
    my_dict['feat_32'].append(list(shape[32]))
    my_dict['feat_34'].append(list(shape[34]))
    my_dict['feat_48'].append(list(shape[48]))
    my_dict['feat_54'].append(list(shape[54]))

    #store the data
    df = read_csv(path_to_csv_key_points)
    add_row(df,my_dict)
    return