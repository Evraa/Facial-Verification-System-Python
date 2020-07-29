import statistics
import pandas
from auxilary import path_to_csv_key_points,np,distance_two_points,read_csv,store_csv

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


# def get_weight():
#     compareFaces()


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



# def store_csv(dataframe, path):
#     '''
#         this function is called implicity to store the new dictionary rows added.
#     '''
#     dataframe.to_csv(path, index=False)

'''
store_csv(calc_weights(), path_to_weighted_set_data)
    From EV:
        + you can call these function in the main file 'main.py' by simply importing them first
        + if u do it like this, it may rasie a problem when calling the file from main.
        + the functions: [store_csv, compareFaces] both exist in auxiliary file
            so either you change their names (if they are doing something different)
            or modify those in auxilary file
            or just import/call those at the auxilary file.
        + when u import functions from auxiliary, you better not type this
        from auxilary import * -> it raises so many warnings
        + better use this
        from auxilary import path_to_csv_key_points,np,distance_two_points,read_csv,store_csv,pandas
        + or this
        import auxilary
            then auxilary.funcName()
        
        + don't forget to doument your functions for further maintenance
        + and erase the TODO comments
        + pandas is already imported in auxilary, re-importing it may occur a time hazard...
        + finally, since the file name is calc_weights, if the file is only doing one job (like this one),
            so its better that the function calc_weights to be the main function that will be called from the main.py
            not store_scv()....
        
        + these constants
        data = read_csv(fileName=path_to_csv_key_points)
        base = data["base_point"]
        features = data[data.columns[3:15]]

        it's better called from/within a function
        because when the main import this file, these constants will also be global
        and this will cause a problem depeneding on the order of importing.
        so the var data may be missed...or filled with undesred data.
'''


'''
    Extra :')
    + the code below, it's better be typed in a file, whether it's [calc_weights, identify_face]
        depending on what it does.



data = read_csv(fileName=path_to_csv_key_points)
def display_weights():
    set_number = int(data.loc[data[data["image_name"] == img].index, "image_set"])
    print("Image", img, "is in set", set_number)
    features = calc_weights().columns.tolist()
    print("Features: ", *features)
    while True:
        feature = input('Enter a feature: ')
        if feature not in features:
            print('Please enter a valid feature')
            continue
        else:
            break
    print("Weight: ", calc_weights().loc[set_number, feature])

# takes in an image name and returns all images of the same face
def like_images(img):
    # indx, rw, d, features, x_scale, threshold_isSame, threshold_isSimilar
    indx = data[data["image_name"] == img].index[0]
    rw = data.loc[indx]
    d = data
    features = data.columns[1:8]
    x_scale = data.iloc[indx, 8]
    threshold_isSame = 5
    threshold_isSimilar = 11
    result = compareFaces(indx, rw, d, features, x_scale, threshold_isSimilar, threshold_isSame)
    return result


# select an image
list_images = data["image_name"].tolist()
while True:
    img = input("Please enter an image: ")
    if img not in list_images:
        print("Sorry, your response must not be negative.")
        continue
    else:
        action = int(input("Would you like to\n[1]: Calculate feature weights\n[2]: Find like images?\n"))
        if action == 1:
            display_weights()
            break
        elif action == 2:
            like_images(img)
            break
        else:
            continue
        break
'''

