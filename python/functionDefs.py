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


# print(CORA_PATH + "/adjList.dict")
adjacency_list_dict = pickle_read(os.path.join(CORA_PATH, "adjList.dict"))
print(adjacency_list_dict)

print("Python script ended")
