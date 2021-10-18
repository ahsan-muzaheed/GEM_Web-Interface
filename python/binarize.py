import pickle
import os
import numpy as np
import sys
from sklearn.preprocessing import label_binarize
from functionDefs import *

# All Cora data is stored as pickle
def pickle_read(path):
    with open(path, "rb") as file:
        data = pickle.load(file)

    return data


def pickle_save(path, data):
    with open(path, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def stringToInt(listInput):
    for i in range(len(listInput)):
        listInput[i] = int(listInput[i])
    return listInput


def countLabel(listInput):
    numOfOne = 0
    numOfZero = 0
    for x in data:
        if x == 1:
            numOfOne += 1
        if x == 0:
            numOfZero += 1
    print("numOfOne: ", numOfOne)
    print("numOfZero: ", numOfZero)


path = os.getcwd() + "/uploads/labels.npy"

data = pickle_read(path)
posVal = sys.argv[1]
posVal = stringToInt(posVal.split(","))
print("Binarized based on: ", posVal)
# Apply positive label
for x in range(len(data)):

    if data[x] in posVal:
        data[x] = 1
    else:
        data[x] = 0

pickle_save(path, data)
countLabel(data)

# newLabel = pickle_read(path)
# print(newLabel)
