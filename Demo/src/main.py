from auxilary import *
# from facial_landmarks import *
# from get_lengths import *
from identify_faces import *
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
def like_images():
    pass


if __name__ == "__main__":
    print("hello :D")
    # TODO: Ev: add a pretty CLI for users

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
                like_images()
                break
            else:
                continue
            break

