import pandas
from sklearn import svm

from auxilary import read_csv, path_to_csv_key_points, distance_two_points

# find two images and train them to be from the same set of in different sets
data = read_csv(fileName=path_to_csv_key_points)
base = data["base_point"]
features = data[data.columns[3:15]]


# for each image set
def get_distances(set_num):
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
        if data.loc[i, "image_set"] == set_num:
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
    print(dists)
    return dists


x = get_distances(1)
x1 = x.loc[1].tolist()
x2 = x.loc[2].tolist()

# two two images to compare
# takes in two arrays of feature lengths/distances
X = [x1, x2]

# labels:
# 0: not the same
# 1: the same
y = [1, 1]
clf = svm.SVC()
clf.fit(X, y)

# print("predict: ", clf.predict([[-1, 3]]))

# get support vectors
print("vectors: ", clf.support_vectors_)

# get indices of support vectors
print("vector indices: ", clf.support_)

# get number of support vectors for each class
print("num vectors: ", clf.n_support_)
