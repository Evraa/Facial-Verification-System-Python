import numpy as np
import os, math
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
from matplotlib import animation
from matplotlib.widgets import Button

import facial_landmarks
import auxilary


def show_image(title, images_list):
    '''
        Simply takes an array of images and show them all together labeld
    '''
    images_list_read = []
    for image_path in images_list:
        img = mpimg.imread(image_path)
        images_list_read.append(img)
    list_length = len(images_list)
    size = math.ceil(math.sqrt(list_length))
    _, axs = plt.subplots(size, size)
    axs = axs.flatten()
    for img, ax in zip(images_list_read, axs):
        ax.imshow(img)
        # ax.set_title(image_name)
        ax.axis('off')

    for i in range(len(images_list), len(axs)):
        axs[i].axis('off')
    plt.suptitle(title, fontsize=14)
    plt.show


def random_numbers(L, H):
    return np.random.random_integers(L, H)


def get_random_image_path(dataset_path):
    # yale set
    folder_numb = random_numbers(1, 15)
    file_numb = random_numbers(0, 10)

    # prof set
    # folder_numb = random_numbers(1, 13)
    # file_numb = 1

    folder_path = dataset_path + str(folder_numb) + '/'
    images = os.listdir(folder_path)
    image_name = images[file_numb]
    image_path = folder_path + image_name
    if not os.path.exists(image_path):
        print(f'ERROR: That image path {image_path} DOES NOT exist')
        return False
    return image_path, image_name


def show_one_image(orig_image_path):
    img = mpimg.imread(orig_image_path)
    plt.imshow(img)
    plt.show()


def mse_diff(diff_1, diff_2):
    diff_1 = np.array(diff_1)
    diff_2 = np.array(diff_2)

    return abs(np.subtract(diff_1, diff_2))


def add_titlebox(ax, text):
    ax.text(.55, .8, text,
            horizontalalignment='center',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.6),
            fontsize=12.5)
    return ax


def show_tests(dataset_path, clf, detector, predictor):
    '''
        + Takes the folder of sets of images
        + Randomly choose an image, and show [Identicalls, simillars, Fasle call]

        Now that we have an image, let's compare it to every other image in the dataset
        1- get facials of the mian image...then lengths
        2- for each image in the data set other than the original
            1- get facials
            2- calc distances

            3- clac differences with the original
            4- show it to the clf

            5- state the prediction
    '''
    orig_image_path, orig_image_name = get_random_image_path(dataset_path)
    print(f'image name: {orig_image_name}')
    # orig_image_path = '../dataset/yalefaces/6/subject06_0.gif'
    # orig_image_name = "subject01_0.gif"
    orig_key_points, state = facial_landmarks.get_key_points(orig_image_path, detector, predictor)
    while state is False:
        orig_image_path, orig_image_name = get_random_image_path(dataset_path)
        orig_key_points, state = facial_landmarks.get_key_points(orig_image_path, detector, predictor)
    orig_base_point = orig_key_points[auxilary.fixed_key_point]
    orig_key_points = orig_key_points[auxilary.dominant_key_points]
    orig_lengths = auxilary.calc_lengths(orig_key_points, orig_base_point)

    # yale set: remove [:1]
    folder_names = auxilary.mylistdir(dataset_path)
    # [1:]
    # folder_names = ['4']
    identicalls = []
    identicalls_names = []
    similars = []
    similars_names = []
    left_overs = []
    identical_title = "Identical Images"
    similar_title = "Similar Images"

    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    # ims = []

    # show_one_image(orig_image_path,orig_image_name)
    for folder_name in folder_names:
        folder_path = dataset_path + folder_name + '/'
        image_names = os.listdir(folder_path)
        for image_name in image_names:
            str = f'Testing folder {folder_name}, image {image_name}'
            print(str)
            image_path = folder_path + image_name
            # im = ax2.imshow(mpimg.imread(image_path), animated=True)
            # ax2.set_title("Testing...")
            # ax2.axis("off")
            # ims.append([im])

            if image_path == orig_image_path:
                continue
            key_points, state = facial_landmarks.get_key_points(image_path, detector, predictor)
            if state is False:
                continue
            base_point = key_points[auxilary.fixed_key_point]
            key_points = key_points[auxilary.dominant_key_points]
            lengths = auxilary.calc_lengths(key_points, base_point)

            # Calc differences
            diff_input = mse_diff(lengths, orig_lengths)
            # show it to clf
            diff_inputs = []
            diff_inputs.append(diff_input)
            prob = clf.predict_proba(diff_inputs)
            prob = prob[0][1]
            if prob <= 0.5:
                continue
            thsh = 0.99
            if thsh >= prob > 0.95:
                similars.append(image_path)
                similars_names.append(image_name)
                continue
            if prob > thsh:
                identicalls.append(image_path)
                identicalls_names.append(image_name)
                continue
            else:
                left_overs.append(image_path)

    # ax1.imshow(mpimg.imread(orig_image_path))
    # ax1.set_title("Original Photo")
    # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
    #                                 repeat=False)
    plt.show()

    if len(similars) == 0 and len(identicalls) == 0:
        print("No matchings at all")

    # display_sets(identicalls, orig_image_path, identical_title)
    # plt.show()
    # display_sets(similars, orig_image_path, similar_title)
    # plt.show()
    # display_sets(left_overs, orig_image_path, "Non-Matched Images")
    # plt.show()

    buttons(identicalls, similars, left_overs, orig_image_path, identical_title, similar_title, "Non-Matched Images")


def display_sets(img_list, orig, title):
    if img_list:
        get_length = math.ceil(math.sqrt(len(img_list)))
        iden = get_length
        gridsize = (iden, iden * 2)
        fig = plt.figure(figsize=(12, 8))
        ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=get_length, rowspan=get_length)
        ax1.set_title("Original Photo")
        # pos = ax1.get_position()
        # x = pos.x0 + pos.x1 / 3
        # y = pos.y1
        # plt.figtext(x, y, 'Original Photo')
        # plt.figtext(x * 2.2, y, title)
        ax1.imshow(mpimg.imread(orig))
        loc = []
        for row in range(iden):
            for col in range(iden, iden * 2):
                loc.append([row, col])
        for location, img in zip(loc, img_list):
            axn = plt.subplot2grid(gridsize, location)
            axn.axis("off")
            axn.set_title(get_answer(img, orig))
            axn.imshow(mpimg.imread(img))
        fig.suptitle(title, fontsize=16)
    else:
        plt.title("No " + title)
    # plt.show()


def get_answer(new_img, original_img):
    if original_img.split("/")[3] == new_img.split("/")[3]:
        return "M"
    elif is_similar(new_img, original_img):
        return "S"
    else:
        return "NM"

def is_similar(img, orig):
    THRESHOLD = 2
    data = auxilary.read_csv(auxilary.path_to_csv_key_points)
    def img_info(imag):
        dists = []
        name = imag.split("/")[4]
        index = data.index[data["image_name"] == name]
        base_point = [int(i) for i in [x.strip("[]") for x in data[data.columns[2]][index].tolist()[0].split(',')]]
        for f in data[data.columns[3:15]]:
            x = [int(i) for i in [x.strip('[]') for x in data.loc[index, f].tolist()[0].split(',')]]
            dists.append(auxilary.distance_two_points(base_point, x))
        return dists
    li = []
    for x, y in zip(img_info(img), img_info(orig)):
        li.append(abs(x-y))
    return np.average(li) < THRESHOLD

def buttons(identicalls, similars, left_overs, orig_image_path, title1, title2, title3):
    class Index(object):

        plt.imshow(mpimg.imread(orig_image_path))

        def same(self, event):
            display_sets(identicalls, orig_image_path, title1)
            plt.show()

        def similar(self, event):
            display_sets(similars, orig_image_path, title2)
            plt.show()

        def rest(self, event):
            display_sets(left_overs, orig_image_path, title3)
            plt.show()

    callback = Index()
    axsame = plt.axes([0.36, 0, 0.1, 0.075])
    axsimilar = plt.axes([0.47, 0, 0.1, 0.075])
    axrest = plt.axes([0.58, 0, 0.1, 0.075])
    bsame = Button(axsame, 'Identical')
    bsame.on_clicked(callback.same)
    bsimilar = Button(axsimilar, 'Similar')
    bsimilar.on_clicked(callback.similar)
    brest = Button(axrest, 'Rest')
    brest.on_clicked(callback.rest)

    plt.show()