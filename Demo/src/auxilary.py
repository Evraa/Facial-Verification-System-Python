import pandas as pd
import os


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


main_dict = read_csv("csv_example.csv")

features = main_dict.columns[1:8]
x_scale = main_dict.columns[8]
y_scale = main_dict.columns[9]

# append to main dict
df = pd.DataFrame(columns=['Similar to', 'Same as'])

# arbitrary threshold
threshold_isSame = 2
threshold_isSimilar = 10


# compares all faces in the database to one face
# returns two lists: similar to and same as
def compareFaces(indx, rw, d):
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
            r = insertFaces(x)
            if r == 2:
                same.append(d['image_name'][i])
            elif r == 1:
                similar.append(d['image_name'][i])
    pair = [similar, same]
    return pair


# compares all faces in the database to all faces
def permDict(d):
    toAppend = pd.DataFrame({
        'isSimilar': [],
        'isSame': []
    })
    for indx, rw in d.iterrows():
        print("\nAnalyzing Face: ", indx, ", ", d['image_name'][indx])
        toAppend.loc[indx] = compareFaces(indx, rw, d)
    result = pd.concat([d, toAppend], axis=1, sort=False)
    return result


# returns simple difference between ratios
def compareRatios(int1, int2):
    v = abs(int1 - int2)
    return v

#
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
def insertFaces(v):
    if v < threshold_isSimilar:
        if v < threshold_isSame:
            return 2
        else:
            return 1
    else:
        return 0


print(permDict(main_dict))
