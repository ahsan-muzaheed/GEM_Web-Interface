# Save a dictionary into a pickle file.
import pickle
import os
import numpy as np
from sklearn.manifold import spectral_embedding
import sys

# All Cora data is stored as pickle
def pickle_read(path):
    with open(path, "rb") as file:
        data = pickle.load(file)

    return data


def pickle_save(path, data):
    with open(path, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def isolatedNodeDetector(adjacecny):

    # return the index of isolated node
    index = 0
    for r in adjacency:
        sum = 0
        for v in r:
            sum = sum + v
        if sum == 0:
            print(index)
        index = index + 1


dir_path = os.getcwd()
file_path = dir_path + "/python/results/adjmatrix.npy"

adjacency = pickle_read(file_path)

# file_path = dir_path + "/node_labels.npy"
# file_path = dir_path + "/node_features.csr"
# print(file_path)
# testData = {0: [2, 3], 1: [4], 2: [0, 3], 3: [0, 2, 4], 4: [1, 3]}
# data = testData
data = pickle_read(file_path)

spectral = spectral_embedding(adjacency, n_components=int(sys.argv[1]))
print(spectral)
