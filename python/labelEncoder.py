from sklearn import preprocessing
import pickle
import os
import sys
import numpy as np


def pickle_read(path):
    with open(path, "rb") as file:
        data = pickle.load(file)

    return data


def labelEncoder(inputarray, numFeature):
    # Read Label column
    dir_path = os.getcwd()
    labels = pickle_read(dir_path + "/uploads/labels.npy")

    # Create LabelEncoder
    # le = preprocessing.LabelEncoder()
    # le.fit(labels)
    # le.transform(labels)

    combined = np.insert(inputarray, numFeature, labels, axis=1)
    # print("Combined: ", combined)
    return combined
