from turtle import pd

from auxilary import *

# to read data frame
from pandas import read_csv

main_dict = read_csv("csv_example.csv")

# to know how many images u have
print(main_dict.head())

# to find the main col labels
print(main_dict.columns)

# to extract features only
features = main_dict.columns[1:8]
x_scale = main_dict.columns[8]
y_scale = main_dict.columns[9]
print(features)

# to iterate over them
for feature in features:
    print(feature)

# to find image[i]'s feature
for index, row in main_dict.iterrows():
    print("index: ", index)  # like the id of the row
    # print(row) #the entire row
    print(row[features[0]])
    print(row[features[1]])
    print(row[features[2]])
    print(row[features[3]])
    print(row[features[4]])
    print(row[features[5]])
    print(row[features[6]])


df = pd.DataFrame(columns=['Similar to', 'Same as'])
print(df.head())

threshold_same = .01
threshold_similar = .1

def compareRatios(int1, int2):
    v = abs(int1 - int2)
    print("diff: ", v)
    return v


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


def permDict(dict):
    for indx, rw in dict:
        compareFaces(indx, rw)

def insertFaces(dataframe, v, indx):
    if v < threshold_similar:
        if v < threshold_same:
            obj = dataframe.append({'Index' : indx , 'Similar to' : "True", 'Same as' : "False"} , ignore_index=True)
            dataframe.append(main_dict.columns[0])


permDict(main_dict.iterrows())
obj = df.append({'Similar to': "True", 'Same as': "False"}, ignore_index=True)

print(obj)