from auxilary import *
from facial_landmarks import draw_landmarks, draw_parts
# from get_lengths import *
# from identify_faces import *
from calc_weights import calc_weights

'''
    Jobs allowed:
        + show an image with facial landmarks on it (dominance option)
        + show an image with drawings on it
        + store key points
        + store lengths...done and wont need it anymore, but it's still there!
        + calc weights
        
'''
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



if __name__ == "__main__":
    print("hello :D")
    # TODO: Ev: add a pretty CLI for users
    #
    # select an image
    # list_images = data["image_name"].tolist()
    # while True:
    #     img = input("Please enter an image: ")
    #     if img not in list_images:
    #         print("Sorry, your response must not be negative.")
    #         continue
    #     else:
    #         action = int(input("Would you like to\n[1]: Calculate feature weights\n[2]: Find like images?\n"))
    #         if action == 1:
    #             display_weights()
    #             break
    #         elif action == 2:
    #             like_images(img)
    #             break
    #         else:
    #             continue
    #         break

    #start of Evram's code
    image_path = path_to_all_dataset + "Mag.jpg"
    # image_path = path_to_all_dataset + "14.jpg"
    draw_landmarks(image_path, circle_type = "dominant")
    draw_parts(image_path)

