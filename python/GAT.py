import sys
import json
import torch.nn as nn
from torch.optim import Adam

# Main computation libraries
import scipy.sparse as sp
import numpy as np

# Deep learning related imports
import torch
import networkx as nx
import os
import pickle
import time
from GATClass import GAT
from LoopPhaseClass import *
import re  # regex


def resultWrite(content):
    with open(
        os.path.join(os.getcwd(), "python", "results", "result.txt"), "a"
    ) as myfile:
        myfile.write(content + "\n")


def stringToInt(listInput):
    for i in range(len(listInput)):
        listInput[i] = int(listInput[i])
    return listInput


# All Cora data is stored as pickle
def pickle_read(path):
    with open(path, "rb") as file:
        data = pickle.load(file)

    return data


def pickle_save(path, data):
    with open(path, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


# Covert JSON to Dictionsary
def jsonToDict(jsonString):

    dictionary = json.loads(jsonString)
    # print(aDict)
    # print(aDict["splitMethod"])
    # print(aDict["range"])

    return dictionary


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


def build_edge_index(adjacency_list_dict, num_of_nodes, add_self_edges=True):
    source_nodes_ids, target_nodes_ids = [], []
    seen_edges = set()

    for src_node, neighboring_nodes in adjacency_list_dict.items():
        for trg_node in neighboring_nodes:
            # if this edge hasn't been seen so far we add it to the edge index (coalescing - removing duplicates)
            if (
                src_node,
                trg_node,
            ) not in seen_edges:  # it'd be easy to explicitly remove self-edges (Cora has none..)
                source_nodes_ids.append(src_node)
                target_nodes_ids.append(trg_node)

                seen_edges.add((src_node, trg_node))

    if add_self_edges:
        source_nodes_ids.extend(np.arange(num_of_nodes))
        target_nodes_ids.extend(np.arange(num_of_nodes))

    # shape = (2, E), where E is the number of edges in the graph
    edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))

    return edge_index


def get_main_loop(
    config,
    gat,
    cross_entropy_loss,
    optimizer,
    node_features,
    node_labels,
    edge_index,
    train_indices,
    val_indices,
    test_indices,
    patience_period,
    time_start,
):
    node_dim = 0  # this will likely change as soon as I add an inductive example (Cora is transductive)

    train_labels = node_labels.index_select(node_dim, train_indices)
    val_labels = node_labels.index_select(node_dim, val_indices)
    test_labels = node_labels.index_select(node_dim, test_indices)

    # node_features shape = (N, FIN), edge_index shape = (2, E)
    graph_data = (
        node_features,
        edge_index,
    )  # I pack data into tuples because GAT uses nn.Sequential which requires it

    def get_node_indices(phase):
        if phase == LoopPhase.TRAIN:
            return train_indices
        elif phase == LoopPhase.VAL:
            return val_indices
        else:
            return test_indices

    def get_node_labels(phase):
        if phase == LoopPhase.TRAIN:
            return train_labels
        elif phase == LoopPhase.VAL:
            return val_labels
        else:
            return test_labels

    def get_training_state(training_config, model):
        training_state = {
            # Training details
            "num_of_epochs": training_config["num_of_epochs"],
            "test_acc": training_config["test_acc"],
            # Model structure
            "num_of_layers": training_config["num_of_layers"],
            "num_heads_per_layer": training_config["num_heads_per_layer"],
            "num_features_per_layer": training_config["num_features_per_layer"],
            "add_skip_connection": training_config["add_skip_connection"],
            "bias": training_config["bias"],
            "dropout": training_config["dropout"],
            # Model state
            "state_dict": model.state_dict(),
        }

        return training_state

    def main_loop(phase, epoch=0):

        global BEST_VAL_ACC, BEST_VAL_LOSS, PATIENCE_CNT, writer

        # Certain modules behave differently depending on whether we're training the model or not.
        # e.g. nn.Dropout - we only want to drop model weights during the training.
        if phase == LoopPhase.TRAIN:
            gat.train()
        else:
            gat.eval()

        node_indices = get_node_indices(phase)
        gt_node_labels = get_node_labels(phase)  # gt stands for ground truth

        # Do a forwards pass and extract only the relevant node scores (train/val or test ones)
        # Note: [0] just extracts the node_features part of the data (index 1 contains the edge_index)
        # shape = (N, C) where N is the number of nodes in the split (train/val/test) and C is the number of classes
        nodes_unnormalized_scores = gat(graph_data)[0].index_select(
            node_dim, node_indices
        )

        # Example: let's take an output for a single node on Cora - it's a vector of size 7 and it contains unnormalized
        # scores like: V = [-1.393,  3.0765, -2.4445,  9.6219,  2.1658, -5.5243, -4.6247]
        # What PyTorch's cross entropy loss does is for every such vector it first applies a softmax, and so we'll
        # have the V transformed into: [1.6421e-05, 1.4338e-03, 5.7378e-06, 0.99797, 5.7673e-04, 2.6376e-07, 6.4848e-07]
        # secondly, whatever the correct class is (say it's 3), it will then take the element at position 3,
        # 0.99797 in this case, and the loss will be -log(0.99797). It does this for every node and applies a mean.
        # You can see that as the probability of the correct class for most nodes approaches 1 we get to 0 loss! <3
        loss = cross_entropy_loss(nodes_unnormalized_scores, gt_node_labels)

        if phase == LoopPhase.TRAIN:
            optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph (.grad fields)
            loss.backward()  # compute the gradients for every trainable weight in the computational graph
            optimizer.step()  # apply the gradients to weights

        # Finds the index of maximum (unnormalized) score for every node and that's the class prediction for that node.
        # Compare those to true (ground truth) labels and find the fraction of correct predictions -> accuracy metric.
        class_predictions = torch.argmax(nodes_unnormalized_scores, dim=-1)
        accuracy = torch.sum(
            torch.eq(class_predictions, gt_node_labels).long()
        ).item() / len(gt_node_labels)

        #
        # Logging
        #

        if phase == LoopPhase.TRAIN:

            # Log metrics
            if True:
                writer.add_scalar("training_loss", loss.item(), epoch)
                writer.add_scalar("training_acc", accuracy, epoch)

            # Save model checkpoint
            if (
                config["checkpoint_freq"] is not None
                and (epoch + 1) % config["checkpoint_freq"] == 0
            ):
                ckpt_model_name = f"gat_ckpt_epoch_{epoch + 1}.pth"
                config["test_acc"] = -1
                torch.save(
                    get_training_state(config, gat),
                    os.path.join(CHECKPOINTS_PATH, ckpt_model_name),
                )

        elif phase == LoopPhase.VAL:
            # Log metrics
            if config["enable_tensorboard"]:
                writer.add_scalar("val_loss", loss.item(), epoch)
                writer.add_scalar("val_acc", accuracy, epoch)

            # Log to console
            if (
                config["console_log_freq"] is not None
                and epoch % config["console_log_freq"] == 0
            ):
                print(
                    f"GAT training: time elapsed= {(time.time() - time_start):.2f} [s] | epoch={epoch + 1} | val acc={accuracy}"
                )
                resultWrite(
                    f"GAT training: time elapsed= {(time.time() - time_start):.2f} [s] | epoch={epoch + 1} | val acc={accuracy}"
                )

            # The "patience" logic - should we break out from the training loop? If either validation acc keeps going up
            # or the val loss keeps going down we won't stop
            if accuracy > BEST_VAL_ACC or loss.item() < BEST_VAL_LOSS:
                BEST_VAL_ACC = max(
                    accuracy, BEST_VAL_ACC
                )  # keep track of the best validation accuracy so far
                BEST_VAL_LOSS = min(loss.item(), BEST_VAL_LOSS)
                PATIENCE_CNT = (
                    0  # reset the counter every time we encounter new best accuracy
                )
            else:
                PATIENCE_CNT += 1  # otherwise keep counting

            if PATIENCE_CNT >= patience_period:
                raise Exception(
                    "Stopping the training, the universe has no more patience for this training."
                )

        else:
            return accuracy  # in the case of test phase we just report back the test accuracy

    return main_loop  # return the decorated function


def load_graph_data(training_config, device):
    CORA_PATH = os.path.join(os.getcwd(), "uploads")
    CORA_TRAIN_RANGE = training_config["train_range"]
    CORA_VAL_RANGE = training_config["val_range"]
    CORA_TEST_RANGE = training_config["test_range"]

    # shape = (N, FIN), where N is the number of nodes and FIN is the number of input features
    node_features_csr = pickle_read(os.path.join(CORA_PATH, "nodeFeature.csr"))

    # shape = (N, 1)
    node_labels_npy = pickle_read(os.path.join(CORA_PATH, "labels.npy"))
    # shape = (N, number of neighboring nodes) <- this is a dictionary not a matrix!
    adjacency_list_dict = pickle_read(os.path.join(CORA_PATH, "adjList.dict"))

    # Normalize the features (helps with training)
    node_features_csr = normalize_features_sparse(node_features_csr)
    num_of_nodes = len(node_labels_npy)

    # shape = (2, E), where E is the number of edges, and 2 for source and target nodes. Basically edge index
    # contains tuples of the format S->T, e.g. 0->3 means that node with id 0 points to a node with id 3.
    topology = build_edge_index(adjacency_list_dict, num_of_nodes, add_self_edges=True)

    # Note: topology is just a fancy way of naming the graph structure data
    # (aside from edge index it could be in the form of an adjacency matrix)

    # if should_visualize:  # network analysis and graph drawing
    #     plot_in_out_degree_distributions(topology, num_of_nodes, dataset_name)  # we'll define these in a second
    #     visualize_graph(topology, node_labels_npy, dataset_name)

    # Convert to dense PyTorch tensors

    # Needs to be long int type because later functions like PyTorch's index_select expect it
    topology = torch.tensor(topology, dtype=torch.long, device=device)
    node_labels = torch.tensor(
        node_labels_npy, dtype=torch.long, device=device
    )  # Cross entropy expects a long int
    node_features = torch.tensor(node_features_csr.todense(), device=device)

    # Indices that help us extract nodes that belong to the train/val and test splits
    train_indices = torch.arange(
        CORA_TRAIN_RANGE[0], CORA_TRAIN_RANGE[1], dtype=torch.long, device=device
    )
    val_indices = torch.arange(
        CORA_VAL_RANGE[0], CORA_VAL_RANGE[1], dtype=torch.long, device=device
    )
    test_indices = torch.arange(
        CORA_TEST_RANGE[0], CORA_TEST_RANGE[1], dtype=torch.long, device=device
    )

    return (
        node_features,
        node_labels,
        topology,
        train_indices,
        val_indices,
        test_indices,
    )


def train_gat(config):
    global BEST_VAL_ACC, BEST_VAL_LOSS

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # checking whether you have a GPU, I hope so!
    # Step 1: load the graph data
    (
        node_features,
        node_labels,
        edge_index,
        train_indices,
        val_indices,
        test_indices,
    ) = load_graph_data(config, device)

    # Step 2: prepare the model
    gat = GAT(
        num_of_layers=config["num_of_layers"],
        num_heads_per_layer=config["num_heads_per_layer"],
        num_features_per_layer=config["num_features_per_layer"],
        add_skip_connection=config["add_skip_connection"],
        bias=config["bias"],
        dropout=config["dropout"],
        log_attention_weights=False,  # no need to store attentions, used only in playground.py while visualizing
    ).to(device)

    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = Adam(
        gat.parameters(), lr=config["learning_rate"], weight_decay=float(5e-4)
    )

    # THIS IS THE CORE OF THE TRAINING (we'll define it in a minute)
    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    file1 = open(os.path.join(os.getcwd(), "python", "results", "result.txt"), "w")
    main_loop = get_main_loop(
        config,
        gat,
        loss_fn,
        optimizer,
        node_features,
        node_labels,
        edge_index,
        train_indices,
        val_indices,
        test_indices,
        1000,
        time.time(),
    )

    BEST_VAL_ACC, BEST_VAL_LOSS, PATIENCE_CNT = [
        0,
        0,
        0,
    ]  # reset vars used for early stopping

    # Step 4: Start the training procedure
    for epoch in range(config["num_of_epochs"]):

        # Training loop
        main_loop(phase=LoopPhase.TRAIN, epoch=epoch)
        # Validation loop
        with torch.no_grad():
            try:
                main_loop(phase=LoopPhase.VAL, epoch=epoch)
            except Exception as e:  # "patience has run out" exception :O
                print(str(e))
                break  # break out from the training loop

    # Step 5: Potentially test your model
    # Don't overfit to the test dataset - only when you've fine-tuned your model on the validation dataset should you
    # report your final loss and accuracy on the test dataset. Friends don't let friends overfit to the test data. <3
    if config["should_test"]:
        test_acc = main_loop(phase=LoopPhase.TEST)
        config["test_acc"] = test_acc
        print(f"Test accuracy = {test_acc}")
        resultWrite(f"Test accuracy = {test_acc}")
    else:
        config["test_acc"] = -1

    # Save the latest GAT in the binaries directory
    torch.save(
        get_training_state(config, gat),
        os.path.join(BINARIES_PATH, get_available_binary_name()),
    )


# This one makes sure we don't overwrite the valuable model binaries (feel free to ignore - not crucial to GAT method)
def get_available_binary_name():
    prefix = "gat"

    def valid_binary_name(binary_name):
        # First time you see raw f-string? Don't worry the only trick is to double the brackets.
        pattern = re.compile(rf"{prefix}_[0-9]{{6}}\.pth")
        return re.fullmatch(pattern, binary_name) is not None

    # Just list the existing binaries so that we don't overwrite them but write to a new one
    valid_binary_names = list(filter(valid_binary_name, os.listdir(BINARIES_PATH)))
    if len(valid_binary_names) > 0:
        last_binary_name = sorted(valid_binary_names)[-1]
        new_suffix = int(last_binary_name.split(".")[0][-6:]) + 1  # increment by 1
        return f"{prefix}_{str(new_suffix).zfill(6)}.pth"
    else:
        return f"{prefix}_000000.pth"


def get_training_state(training_config, model):
    training_state = {
        # Training details
        "num_of_epochs": training_config["num_of_epochs"],
        "test_acc": training_config["test_acc"],
        # Model structure
        "num_of_layers": training_config["num_of_layers"],
        "num_heads_per_layer": training_config["num_heads_per_layer"],
        "num_features_per_layer": training_config["num_features_per_layer"],
        "add_skip_connection": training_config["add_skip_connection"],
        "bias": training_config["bias"],
        "dropout": training_config["dropout"],
        # Model state
        "state_dict": model.state_dict(),
    }

    return training_state


def dictTrimmer(jsonDict):
    # embeddingMethod = jsonDict["embeddingMethod"]
    # num_of_epochs = int(jsonDict["num_of_epochs"])
    # learning_rate = float(jsonDict["learning_rate"])
    # isDirect = jsonDict["isDirect"]
    # num_of_layers = int(jsonDict["num_of_layers"])
    # num_heads_per_layer = jsonDict["num_heads_per_layer"].split(",")
    # num_features_per_layer = jsonDict["num_features_per_layer"].split(",")
    # add_skip_connection = eval(jsonDict["add_skip_connection"])
    # bias = eval(jsonDict["bias"])
    # dropout = float(jsonDict["dropout"])
    # train_range = jsonDict["train_range"].split(",")
    # val_range = jsonDict["val_range"].split(",")
    # test_range = jsonDict["test_range"].split(",")

    embeddingMethod = jsonDict["embeddingMethod"]
    jsonDict["num_of_epochs"] = int(jsonDict["num_of_epochs"])
    jsonDict["learning_rate"] = float(jsonDict["learning_rate"])
    jsonDict["should_test"] = eval(jsonDict["should_test"])
    jsonDict["num_of_layers"] = int(jsonDict["num_of_layers"])
    jsonDict["num_heads_per_layer"] = stringToInt(
        jsonDict["num_heads_per_layer"].split(",")
    )
    jsonDict["num_features_per_layer"] = stringToInt(
        jsonDict["num_features_per_layer"].split(",")
    )
    jsonDict["add_skip_connection"] = eval(jsonDict["add_skip_connection"])
    jsonDict["bias"] = eval(jsonDict["bias"])
    jsonDict["dropout"] = float(jsonDict["dropout"])
    jsonDict["train_range"] = stringToInt(jsonDict["train_range"].split(","))
    jsonDict["val_range"] = stringToInt(jsonDict["val_range"].split(","))
    jsonDict["test_range"] = stringToInt(jsonDict["test_range"].split(","))
    jsonDict["enable_tensorboard"] = False
    jsonDict["console_log_freq"] = 100
    jsonDict["checkpoint_freq"] = 1000
    jsonDict["weight_decay"] = float(5e-4)

    return jsonDict


if __name__ == "__main__":
    jsonDict = jsonToDict(sys.argv[1])
    training_config = dictTrimmer(jsonDict)
    train_gat(training_config)
