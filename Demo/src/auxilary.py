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
threshold_isSame = .01
threshold_isSimilar = .1


# compares all faces in the database to one face
def compareFaces(indx, rw):
    for f in range(0, 6):
        xscale = rw[features[f]] / rw[x_scale]
        print("\nFor ", features[f], " of Index ", indx, " (", rw[features[f]], ")")
        print("Face Scale X: ", rw[x_scale])
        for i, r in main_dict.iterrows():
            if not i == indx:
                xscale2 = r[features[f]] / r[x_scale]
                print("Compare to: ", i, " value: ", r[features[f]], " scale: ", xscale2)
                compareRatios(xscale, xscale2)


# compares all faces in the database to all faces
def permDict(dict):
    for indx, rw in dict:
        compareFaces(indx, rw)


# returns simple difference between ratios
def compareRatios(int1, int2):
    v = abs(int1 - int2)
    print("diff: ", v)
    return v


# inserts the picture name into the new columns depending of threshold
def insertFaces(dataframe, v, indx):
    if v < threshold_isSimilar:
        if v < threshold_isSame:
            obj = dataframe.append({'Index': indx, 'Similar to': "True", 'Same as': "False"}, ignore_index=True)
            dataframe.append(main_dict.columns[0])
