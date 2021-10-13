import pickle

# Visualization related imports
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import networkx as nx

# import igraph as ig

# Main computation libraries
import scipy.sparse as sp
import numpy as np

# Deep learning related imports
import torch

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
# print("Cora Path is: ", CORA_PATH)
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


def ismultiClass(num_of_nodes):
    if num_of_nodes > 2:
        return True
    else:
        return False


def uniqueFromList(listA):
    unique_list = []
    for x in listA:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def countX(x, lst):
    count = 0
    for ele in lst:
        if ele == x:
            count = count + 1
    return count


def normalize_features_sparse(node_features_sparse):
    assert sp.issparse(
        node_features_sparse
    ), f"Expected a sparse matrix, got {node_features_sparse}."

    # Instead of dividing (like in normalize_features_dense()) we do multiplication with inverse sum of features.
    # Modern hardware (GPUs, TPUs, ASICs) is optimized for fast matrix multiplications! ^^ (* >> /)
    # shape = (N, FIN) -> (N, 1), where N number of nodes and FIN number of input features
    node_features_sum = np.array(
        node_features_sparse.sum(-1)
    )  # sum features for every node feature vector

    # Make an inverse (remember * by 1/x is better (faster) then / by x)
    # shape = (N, 1) -> (N)
    node_features_inv_sum = np.power(node_features_sum, -1).squeeze()

    # Again certain sums will be 0 so 1/0 will give us inf so we replace those by 1 which is a neutral element for mul
    node_features_inv_sum[np.isinf(node_features_inv_sum)] = 1.0

    # Create a diagonal matrix whose values on the diagonal come from node_features_inv_sum
    diagonal_inv_features_sum_matrix = sp.diags(node_features_inv_sum)

    # We return the normalized features.
    return diagonal_inv_features_sum_matrix.dot(node_features_sparse)


# shape = (N, number of neighboring nodes) <- this is a dictionary not a matrix!
adjacency_list_dict = pickle_read(os.path.join(CORA_PATH, "adjList.dict"))

# {1:[2, 4, 102], 2:[30, 5, 22]}...
# print("Adjacency list: ", adjacency_list_dict)

# shape = (N, FIN), where N is the number of nodes and FIN is the number of input features
node_features_csr = pickle_read(os.path.join(CORA_PATH, "nodeFeature.csr"))
normalized_feature = normalize_features_sparse(node_features_csr)
densed = node_features_csr.todense()

# print(densed[:2, :20])
# (0,81) 1.0   (0,146) 1.0, (1, 687) 1.0  ... (2707, 1414) 1.0
# print("node-feature: ", node_features_csr)


# shape = (N, 1)
node_labels_npy = pickle_read(os.path.join(CORA_PATH, "labels.npy"))

# print(adjacency_list_dict)
unique_label = uniqueFromList(node_labels_npy)
unique_label.sort()
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

# Multiclass detection
if len(unique_label) > 2:
    # file = open(savePath + "/isMulti.txt","w")

    f.write("NumOfClass: " + str(len(unique_label)) + " \n")
    f.write("Label: " + str(unique_label) + "\n")

    for i in unique_label:
        # print(i)
        # print(countX(i, node_labels_npy))
        f.write(str(i) + ": " + str(countX(i, node_labels_npy)) + "\n")


# Result Managing
result1 = "Adjacency Matrix: " + str(num_of_nodes) + "x" + str(num_of_nodes) + "\n"
result2 = str(matrix)
f.write(result1)
f.write(result2)

# Save pickle file for future usage
pickle_save(savePath + "/adjmatrix.npy", matrix)


print("Python script ended")
