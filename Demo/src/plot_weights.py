#GLOBAL IMPORTS
import numpy as np
from glob import glob
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import copy

#LOCAL IMPORTS
import auxilary




def get_diff_similar(csv_path):
    data = auxilary.read_csv(fileName=csv_path)
    N = data.shape[0] #5050
    D = data.shape[1] - 1 #22
    george_data = data[data['output'] == 'George_W_Bush']
    sim_diff = []
    ev = 0
    for index_1, row_1 in george_data.iterrows():
        features_1 = np.array(row_1[:D])
        if ev%50 == 0:
            print (f'image: {ev}')
        ev += 1
        for index_2, row_2 in george_data.iterrows():
            if index_1 >= index_2:
                continue
            features_2 = list(row_2[:D])
            
            sim_diff.append(np.power( (features_1-features_2), 2))

    df = pd.DataFrame(sim_diff)
    df["label"] = 1
    df.to_csv("../csv_files/similars_diff.csv",index=False)

def get_diff_diff (csv_path):
    similars = auxilary.read_csv(fileName="../csv_files/similars_diff.csv")
    iterations =  (len(similars))

    data = auxilary.read_csv(fileName=csv_path)
    N = data.shape[0] #5050
    D = data.shape[1] - 1 #22
    diff_diff = []
    for i in range (iterations):
        if i %1000 == 0:
            print (f'iteration: {i}')
        random_int = np.random.randint(0, N)
        row_1 = data.iloc[random_int]
        features_1 = np.array(row_1[:D])

        random_int = np.random.randint(0, N)
        row_2 = data.iloc[random_int]

        while row_2['output'] == row_1['output']:
            random_int = np.random.randint(0, N)
            row_2 = data.iloc[random_int]
        
        features_2 = np.array(row_2[:D])
        
        diff_diff.append(np.power( (features_1-features_2), 2))

    df = pd.DataFrame(diff_diff)
    df["label"] = 0
    df.to_csv("../csv_files/different_diff.csv",index=False)

def compare ():
    similars    = auxilary.read_csv(fileName="../csv_files/similars_diff.csv")
    differs     = auxilary.read_csv(fileName="../csv_files/different_diff.csv")

    sims = np.array(similars.mean())
    difs = np.array(differs.mean())
    
    sub = np.subtract(difs, sims)
    sub = sub[:-1]
    feats = np.zeros([22, 2])

    for i in range (22):
        if sub[i] > 0:
            feats[i] = [int(i), sub[i]]

    sorted_array = feats[np.argsort(feats[:, 1])]
    flipped = np.flip(sorted_array)
    w_mul = copy.deepcopy(feats)

    feats = flipped[:, 0]
    idx = flipped [:,1]
    summ =  (np.sum(feats))

    w_mul = w_mul[:, 1] / summ

    prob = (feats) / summ 

    return prob, idx, w_mul*100

def run_main(csv_path = "../csv_files/embedded_2.csv"):
    # get_diff_similar(csv_path=csv_path)
    # get_diff_diff (csv_path = csv_path)
    weights, idx, w_mul =  compare ()
    print ('Features index: ')
    print (idx)
    print ("Weight: ")
    print (weights*100)
    print ("weights not sorted: ")
    print (w_mul)
    
