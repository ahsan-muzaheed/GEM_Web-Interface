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
    for x in listInput:
        if x == 1:
            numOfOne += 1
        if x == 0:
            numOfZero += 1
    tempoList.append(("numOfOne: " + str(numOfOne)))
    tempoList.append(("numOfZero: " + str(numOfZero)))


def txtAppender(content):
    import os

    targetPath = os.path.join(os.getcwd(), "python/results/binarizedResult.txt")
    file1 = open(targetPath, "w")

    for i in range(len(content)):
        print(content[i])
        file1.write(content[i] + "\n")

    file1.close()


tempoList = []
path = os.getcwd() + "/uploads/labels.npy"

data = pickle_read(path)
posVal = sys.argv[1]
posVal = stringToInt(posVal.split(","))
tempoList.append("Binarized based on: " + str(posVal))
# Apply positive label
for x in range(len(data)):

    if data[x] in posVal:
        data[x] = 1
    else:
        data[x] = 0

pickle_save(path, data)
countLabel(data)

txtAppender(tempoList)


# newLabel = pickle_read(path)
# print(newLabel)
