import random

import auxilary
import face_recognition
import show_tests
import numpy as np
import os
import NN
import cv2
import facial_landmarks


def euclidean(input_1, input_2):
    diff = input_1 - input_2
    power = np.power(diff, 2)
    return np.sqrt(np.sum(power))


def results(embeddings, inputs, labels):
    identicals = []
    similars = []
    for i, input_elm in enumerate(inputs):
        error = euclidean(embeddings, input_elm)
        if error >= 0.8:
            continue
        if error < 0.8 and error >= 0.75:
            similars.append(labels[i])
        if error < 0.75:
            identicals.append(labels[i])

    return identicals, similars


def trim_outputs(path, labels, face_name, identicals, similars):
    N = labels.shape[0]
    others_names = []
    others_paths = []
    # for others
    for i in range(9):
        rand_int = np.random.random_integers(0, N - 1)
        label = labels[rand_int]
        dir_path = path + str(label) + '/'
        files_count = (auxilary.mylistdir(dir_path))
        while label == face_name or len(files_count) < 1 or \
                label in identicals or label in similars:
            rand_int = np.random.random_integers(0, N - 1)
            label = labels[rand_int]
            dir_path = path + str(label) + '/'
            files_count = (auxilary.mylistdir(dir_path))

        others_names.append(label)
        others_paths.append(dir_path + files_count[0])

    # SIMILARS
    similars_2 = []
    for i in range(len(similars)):
        if len(similars_2) == 9:
            break
        rand_int = np.random.random_integers(0, len(similars))
        sim = similars[i]
        if sim not in similars_2:
            similars_2.append(sim)

    # then, it's good
    sim_paths = []
    sim_names = similars_2
    for sim in similars_2:
        dir_path = path + str(sim) + '/'
        files_count = (auxilary.mylistdir(dir_path))
        sim_paths.append(dir_path + files_count[0])

    # IDENTICALS
    idc_paths, idc_names = [], []
    # make sure the main file occurs
    dir_path = path + face_name + '/'
    files_count = auxilary.mylistdir(dir_path)
    for i, file_count in enumerate(files_count):
        if i >= 9:
            break
        idc_names.append(face_name)
        idc_paths.append(dir_path + file_count)
        if face_name in identicals:
            identicals.remove(face_name)

    others_count = 9 - len(idc_names)

    for i in range(len(identicals)):
        if others_count == 0:
            break
        rand_int = np.random.random_integers(0, len(identicals))
        idc = identicals[i]
        if idc not in idc_names:
            idc_names.append("NOT MATCHING")
            dir_path = path + str(idc) + '/'
            files_count = (auxilary.mylistdir(dir_path))
            idc_paths.append(dir_path + files_count[0])
            others_count -= 1

    print('\n')
    print(idc_names)
    print(idc_paths)
    print("\n\n\n")
    print(sim_names)
    print(sim_paths)
    print("\n\n\n")
    print(others_names)
    print(others_paths)
    print("\n\n\n")
    return idc_paths, idc_names, sim_paths, sim_names, others_paths, others_names


def Euc_result_preview(dataset_path, image_num=None):
    print("Load labels")
    data = auxilary.read_csv(fileName=dataset_path)
    N = len(data)
    D = 22

    data_inputs = (data.iloc[:, :D])
    inputs = np.zeros([N, D])
    inputs = np.array(data_inputs)

    labels = np.zeros([N, 1])
    labels = np.array(data.iloc[:, D])

    target = random.choice(range(N))
    print("Testing image no. ", target, ", ", data.iloc[target, 22])
    embeddings, face_name, human_file_path = face_recognition.face_recognition(dataset_path="../dataset/main_data/*/*",
                                                                               preview=True, image_num=image_num)
    # print(face_name, human_file_path)
    face_name = data.iloc[target, -1]
    embeddings = data.iloc[target, :-1]

    identicals, similars = results(embeddings, inputs, labels)

    # idc_paths, idc_names, sim_paths, sim_names, others_paths, others_names = \
    #     trim_outputs (labels, face_name, identicals, similars)

    all_set = data.loc[data['output'] == target]
    # print(all_set)

    show_tests.buttons(identicalls=identicals, similars=similars,
                       orig_image_path=human_file_path, orig_title=face_name, title1="MATCHING", title2="SIMILARS",
                       title3="OTHERS")


def trim_NN_outputs(path, labels, face_name, identicals, similars, bucket):
    N = labels.shape[0]
    others_names = []
    others_paths = []
    # for others
    for _ in range(9):
        rand_int = np.random.random_integers(0, N - 1)
        label = labels[rand_int]
        dir_path = path + str(label) + '/'
        files_count = (auxilary.mylistdir(dir_path))
        while label == face_name or len(files_count) < 1 or \
                label in identicals or label in similars:
            rand_int = np.random.random_integers(0, N - 1)
            label = labels[rand_int]
            dir_path = path + str(label) + '/'
            files_count = (auxilary.mylistdir(dir_path))

        others_names.append(label)
        others_paths.append(dir_path + files_count[0])

    # for similars
    sim_paths = []
    sim_names = similars

    for sim in similars:
        sim_path = path + str(sim) + '/'
        files = auxilary.mylistdir(sim_path)
        if len(files) > 0:
            sim_paths.append(sim_path + files[0])

    # for identicals
    idc_name = identicals[0]
    idc_paths, idc_names = [], []
    id_path = path + str(idc_name) + '/'
    files = auxilary.mylistdir(id_path)
    count = len(files)
    # it = 9 if count >=9  else count
    for i in range(count):
        idc_names.append(idc_name)
        img_path = id_path + files[i]
        idc_paths.append(img_path)

    return idc_paths, idc_names, sim_paths, sim_names, others_paths, others_names


def NN_result_preview(embedded_data, file_paths, image_path=None, blur=False, pred=None, detc=None, bucket=False):
    print("Load labels")
    data = auxilary.read_csv(fileName=embedded_data)
    D = 22
    N = len(data)
    recur_paths = file_paths+'*/*'

    data_inputs = (data.iloc[:, :D])
    inputs = np.zeros([N, D])
    inputs = np.array(data_inputs)

    labels = np.zeros([N, 1])
    labels = np.array(data.iloc[:, D])

    embeddings, face_name, human_file_path = facial_landmarks.test_preview(
        blur=blur, dataset_path=recur_paths, pred=pred, detc=detc, image_path=image_path)

    # stats = get_stats(data, embeddings)
    stats = None

    # identicals, similars = NN_results(embeddings, inputs, labels)
    identicals, similars = NN.predict_input(embedded_data, embeddings)

    idc_paths, idc_names, sim_paths, sim_names, others_paths, others_names = \
        trim_NN_outputs(file_paths, labels, face_name, identicals, similars, bucket)

    if bucket:
        return idc_paths, stats
    else:
        show_tests.buttons(identicalls=idc_paths, id_titles=idc_names, similars=sim_paths, sim_titles=sim_names,
                           left_overs=others_paths, left_titles=others_names,
                           orig_image_path=human_file_path, orig_title=face_name, title1="Matched", title2="Similar",
                           title3="Others", stats=stats)


def get_stats(data, embeddings):
    av = data.mean()
    variation = (abs(av - embeddings) / av) * 100
    eyebr = np.average(variation[0:5])
    eye = np.average(variation[6:13])
    nose = np.average(variation[14:16])
    mouth = np.average(variation[17:21])
    return {'Eyebrows': eyebr, 'Eyes': eye, "Nose": nose, "Mouth": mouth}
