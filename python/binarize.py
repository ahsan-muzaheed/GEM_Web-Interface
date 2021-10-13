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


path = os.getcwd() + "/uploads/labels.npy"

data = pickle_read(path)
posVal = sys.argv[1]

# Apply positive label
for x in range(len(data)):
    if data[x] == int(posVal):
        data[x] = 1
    else:
        data[x] = 0
print(data[:300])
pickle_save(path, data)


# newLabel = pickle_read(path)
# print(newLabel)
