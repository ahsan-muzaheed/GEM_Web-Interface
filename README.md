# GEM_Web-Interface
This repo contains a Node.js - Python implementation of Graph Embedding Methods(GEM) that allows users to experiment GEM algorithms and downstream machine learning tasks in a Web environment.

## Table of contents
* [What is Graph Embedding Methods (GEM)?](#What-is-Graph-Embedding-Methods-(GEM)?)
* [Setup](#Setup)
* [Usage](#usage)
* [Future Work](#future-work)
## What is Graph Embedding Methods (GEM)?
### Introduction
Everything that consists of nodes and edges in relationship can be considered as a graph, such as network data, DNA, brain networks, etc. Since we can store various non-linear information in graphs, there have been research to apply machine learning algorithms on graphs.
However, it has been very challenging to achieve high-performance from machine learning on graphs due to the scalability issues.

### Representation Learning

Recently, <em>Representation Learning</em>, a way to map the nodes of graphs or entire graph onto the low-dimensional vector space, has been suggested to solve thie issues. We then optimized embedded vectors in the low-dimensional space and use them as input features for downstream machine learning tasks such as node classification, link prediction, etc.


### Graph Embedding Methods

The GEM is about approaches to transform the graph data into vectors in latent space, preserving the topological information from the original graph.
<p align="center">
<img src="readmepics/embedding.png"/>

<em> Representation Learning: [Original image](https://arxiv.org/pdf/1909.00958v1.pdf)</em>


</p>


## Setup
1. `git clone https://github.com/JunyongLee1217/GEM_Web-Interface`
2. Make sure you have recent version of Node.js and Python3
3. `pip3 install -r requirements.txt`
4. `npm init`
5. `npm install`
6. `npm start`

## Test Dataset

Cora dataset is one of 'Hello World' datasets in graph machine learning. The data can be found in 'inputData' folder.


|Nodes|2708|
|------|---|
|Classes|7|
|Citation Links|5429|
|Unique Words|1433|

* adjacecny_list.dict
* node_features.csr
* node_labels.npy
  

## Usage

### Landing Page
The landing page manages the file uploads. It take three files as input.
* Adjacecny List.dict
  <p>Dictionary file containing information about connected nodes
* Node Features.csr
  <p> Feature vector matrix in csr(Compressed Sparse Matrix)
* Labels.npy

<p align="center">
<img src= "readmepics/2LandingPage.png" style=;>
</p>

---

### UploadResult
The upload page presents the result of brief analysis of input files, such as the number of classes, the number of data that each label has, etc.
You can binarize(1 and 0) your data labels if input data has multiple classes. If this the case, the positive labels should be populated in comma separated format.
<p>
<img src="readmepics/3UploadResult.png">
</p>


---

### Graph Embedding

You can choose the preferred graph embedding methods that you need. 

* [Laplacian Eigenmaps](https://proceedings.neurips.cc/paper/2001/file/f106b7f99d2cb30c3db1c3cc0fde9ccb-Paper.pdf)
* [Graph Attention Networks(GAT)](https://arxiv.org/pdf/1710.10903.pdf) 
(re-implentation of [this](https://github.com/gordicaleksa/pytorch-GAT) repository)
* Locally Linear Embeeding(LLE)
* Graph Factorization
* HOPE
* SDNE
* node2vec


So far, Laplacian Eigenmaps and GAT have been integrated. For GAT, it currently provides direct classification. The 'only for embedding' option is under development.

--------------------------------

### Downstrem ML

The downstrem ML page provides machine learning algorithm options that we may apply to the embedded vectors from the graph embedding page.

<p>
<img src="readmepics/8Downstream.png">
</p>

>Supervised Learning
* [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
* [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
* [Decision Tree](https://scikit-learn.org/stable/modules/tree.html)
* Split Method
  - Holdout
  - [K-fold Validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
  - Random Sampling (Taking Average of num of trials)
* Metric
  - [Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
  - [Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
  - [Recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
  - [F1 Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
  - [ROCAUC curve (only for binarized dataset)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
> Unsupervised learning
* [K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
* [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
* [Agglomerative](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
* Metric
  - [Rand Index](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)
  - [Normalized Mutual Information(NMI)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html)
  - [Silhouette Coefficient](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)


### Result Page
The result page shows the results from the metrics chosen in the downstrem ML step including ROCAUC curve. 
<p>
<img src="readmepics/Result.png">
</p>


### Result from GAT
In the case of GAT with direct classification, the result page skips downstream ML step and direclty yields result from each epoch and final accuaracy score.

<p> 
<img src="readmepics/13GATResult.png">
</p>



## Future Work
* Integrating more Graph Embedding algorithms
* GAT's 'only embedding' option
* Interface for taking multiple downsteam ML algorithms on embedded vectors
  