import pickle

# Visualization related imports
import matplotlib.pyplot as plt
import networkx as nx

# import igraph as ig

# Main computation libraries
import scipy.sparse as sp
import numpy as np

# Deep learning related imports
import torch
import os
import enum

# Path for fetching data

DATA_DIR_PATH = os.path.join(os.getcwd())

CORA_PATH = os.path.join(DATA_DIR_PATH, "uploads")
print("Cora Path is: ", CORA_PATH)
CORA_TRAIN_RANGE = [0, 140]  # we're using the first 140 nodes as the training nodes
CORA_VAL_RANGE = [140, 140 + 500]
CORA_TEST_RANGE = [1708, 1708 + 1000]
CORA_NUM_INPUT_FEATURES = 1433
CORA_NUM_CLASSES = 7
# Used whenever we need to visualzie points from different classes (t-SNE, CORA visualization)
cora_label_to_color_map = {
    0: "red",
    1: "blue",
    2: "green",
    3: "orange",
    4: "yellow",
    5: "pink",
    6: "gray",
}


# All Cora data is stored as pickle
def pickle_read(path):
    with open(path, "rb") as file:
        data = pickle.load(file)

    return data


def pickle_save(path, data):
    with open(path, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def adjMatrixBuilder(data):
    keys = sorted(data.keys())
    size = len(keys)
    matrix = np.zeros((size, size))
    for k, vs in data.items():
        for v in vs:

            if v in data[k]:

                matrix[k][v] = 1
    return matrix, size


# shape = (N, number of neighboring nodes) <- this is a dictionary not a matrix!
adjacency_list_dict = pickle_read(os.path.join(CORA_PATH, "adjList.dict"))
# shape = (N, FIN), where N is the number of nodes and FIN is the number of input features
node_features_csr = pickle_read(os.path.join(CORA_PATH, "nodeFeature.csr"))
# shape = (N, 1)
node_labels_npy = pickle_read(os.path.join(CORA_PATH, "labels.npy"))
# print(type(node_labels_npy))
# Normalize the features (helps with training)
# node_features_csr = normalize_features_sparse(node_features_csr)


if len(node_labels_npy) > 0:
    num_of_nodes = len(node_labels_npy)
if adjacency_list_dict:
    matrix, num_of_nodes = adjMatrixBuilder(adjacency_list_dict)

# Console Printing
# print("Adjacency Matrix: ", num_of_nodes, "x", num_of_nodes)
# print("Matrix is: ", matrix)

# File Writing
savePath = os.path.join(DATA_DIR_PATH, "python/results")
f = open(savePath + "/num_of_nodes.txt", "w")

# Result Managing
result1 = "Adjacency Matrix: " + str(num_of_nodes) + "x" + str(num_of_nodes) + "\n"
result2 = str(matrix)
f.write(result1)
f.write(result2)

# Save pickle file for future usage
pickle_save(savePath + "/adjmatrix.npy", matrix)


print("Python script ended")
