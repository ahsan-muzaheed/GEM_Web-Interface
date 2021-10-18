import json
import os
import numpy as np
from metricFuncs import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import explained_variance_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from numpy import mean
from numpy import std
from txtHandler import *


# import pandas as pd
import sys


# Covert JSON to Dictionsary
def jsonToDict(jsonString):

    dictionary = json.loads(jsonString)
    # print(aDict)
    # print(aDict["splitMethod"])
    # print(aDict["range"])

    return dictionary


def readData():
    targetPath = os.path.join(os.getcwd(), "python/results/dataSet.npy")
    data = np.load(targetPath, allow_pickle=True)

    return data


def dataSplitter(ratioInt, dataSet):

    trainRatio = ratioInt * 0.01
    X = dataSet[:, :-1]
    y = dataSet[:, -1:].ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=trainRatio, shuffle=True
    )

    return X_train, X_test, y_train, y_test


def kfoldHandler(clf, kval, dataSet, metric, posVal):
    # txtAppender("Kvalue for Kfold: " + str(kval))
    stringResultList = []
    stringResultList.append("Output from downstrem ML")
    stringResultList.append("------------------------")
    stringResultList.append("Method: Kfold")
    stringResultList.append("------------------------")
    stringResultList.append("Classifier  : " + str(clf.__class__.__name__))
    stringResultList.append("Kvalue for Kfold: " + str(kval))
    X = dataSet[:, :-1]
    y = dataSet[:, -1:].ravel()
    # y = preprocessing.label_binarize(y, classes=[0, 1])
    # print(y[:500])
    # prepare the cross-validation procedure

    cv = KFold(n_splits=kval, random_state=1, shuffle=True)

    temporalResultList = kfoldMetric(cv, clf, metric, X, y)

    stringResultList = stringResultList + temporalResultList
    # accuracy = cross_val_score(clf, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
    print(stringResultList)
    # report performance

    # print("Accuracy: %.3f (%.3f)" % (mean(accuracy), std(accuracy)))
    # print("precision:", precision)
    # return mean(accuracy), std(accuracy)
    return stringResultList


def classGetter(y):

    return list(set(y.ravel()))


def fprtprCalulator(clf, X_test, y_test):

    r_probs = [0 for _ in range(len(y_test))]
    clf_probs = clf.predict_proba(X_test)

    # Keep positive outcome
    clf_probs = clf_probs[:, 1]

    # Calculate AUROC
    r_auc = roc_auc_score(y_test, r_probs)
    clf_auc = roc_auc_score(y_test, clf_probs)

    # Print AUROC Score
    # print("Random (chance) Prediction: AUROC = %.3f" % (r_auc))
    # print(str(clf.__class__.__name__) + " AUROC = %.3f" % (clf_auc))

    # Calculate ROC curve
    r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
    clf_fpr, clf_tpr, _ = roc_curve(y_test, clf_probs)

    return clf_fpr, clf_tpr, r_fpr, r_tpr, r_auc, clf_auc


def plottingROC(clf, clf_fpr, clf_tpr, r_fpr, r_tpr, r_auc, clf_auc, methodName):

    # Plot the ROC curve
    plt.plot(
        r_fpr,
        r_tpr,
        linestyle="--",
        label="Random Prediction (AUROC = %0.3f)" % (r_auc),
    )
    plt.plot(
        clf_fpr,
        clf_tpr,
        marker=".",
        label=str(clf.__class__.__name__) + " (AUROC = %0.3f" % (clf_auc),
    )

    # title
    plt.title("ROC Curve for " + str(clf.__class__.__name__))

    # Axis labels
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # Show legend

    plt.legend()

    nameTag = 0
    path = (
        os.getcwd()
        + "/python/results/rocCurve/"
        + methodName
        + "_rocCurve_"
        + str(clf.__class__.__name__)
        + ".png"
    )
    plt.savefig(path)

    # plt.show()


def roc_curve_processor(clf, X_test, y_test, methodName):

    fprVal, tprVal, r_fpr, r_tpr, r_auc, clf_auc = fprtprCalulator(clf, X_test, y_test)
    plottingROC(clf, fprVal, tprVal, r_fpr, r_tpr, r_auc, clf_auc, methodName)


def randomSamplingHandler(clf, trainRatio, numOfIter, dataSet, metric, posVal):
    stringResultList = []
    temporalDict = {}
    for i in metric:
        temporalDict[i] = []

    targetColumn = dataSet[:, -1:]
    classList = classGetter(targetColumn)

    y_binarized = label_binarize(
        targetColumn, classes=classList
    )  # Now y is multi dimensional
    n_classes = y_binarized.shape[1]

    # Compute ROC curve and ROC area for each class
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(10, 10))
    stringResultList.append("Output from downstrem ML")
    stringResultList.append("------------------------")
    stringResultList.append("Method: randomSampling")
    stringResultList.append(("------------------------"))
    stringResultList.append(("Classifier  : " + str(clf)))
    stringResultList.append(("------------------------"))
    stringResultList.append(("Num of Trials: " + str(numOfIter)))

    for i in range(numOfIter):

        X_train, X_test, y_train, y_test = dataSplitter(trainRatio, dataSet)
        clf.fit(X_train, y_train)
        stringResultList.append(("Train ratio : " + str(trainRatio) + "% Done."))
        stringResultList.append(("------------------------"))
        y_pred = clf.predict(X_test)

        if "rocauc" in metric:
            methodName = "randomSampling"
            roc_curve_processor(clf, X_test, y_test, methodName)

    temporalList = metricExamine(metric, y_test, y_pred)

    # Accumulate each metric values to dictionary
    for j in range(len(temporalList)):
        temporalDict[metric[j]].append(temporalList[j])

    # print(temporalDict)
    values = temporalDict.values()
    for k, p in zip(values, range(len(metric))):

        temporalDict[metric[p]] = mean(k), std(k)
        # print(type(temporalDict[metric[p]]))

    # print("after:", temporalDict)

    for k, v in temporalDict.items():
        if k != "rocauc":
            stringResultList.append(
                (str(k) + ": " + "Average: " + str(v[0]) + " Std: " + str(v[1]))
            )

    return stringResultList
    # for y in range(2):
    #    print("Average: ", list(temporalDict.values())[y][0])
    #    print("Std: ", list(temporalDict.values())[y][1])

    # acc = metrics.accuracy_score(y_test, y_pred)
    # variance = explained_variance_score(y_test, y_pred)
    # listForAcc.append(acc)

    # Get average and std
    # Accmean = mean(listForAcc)
    # Accstd = std(listForAcc)

    # return Accmean, Accstd


def holdoutHandler(clf, trainRatio, X_train, X_test, y_train, y_test, metric):
    stringResultList = []

    # Result show in console
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    stringResultList.append("Output from downstrem ML")
    stringResultList.append("------------------------")
    stringResultList.append("Method: holdout")
    stringResultList.append(("------------------------"))
    stringResultList.append("Classifier  : " + str(clf))
    stringResultList.append("Train ratio : " + str(trainRatio) + "%")
    stringResultList.append("Num of Train: " + str(len(X_train)))
    stringResultList.append("Num of Test : " + str(len(X_test)))
    stringResultList.append("------------------------")
    # print("y_test: ", y_test)
    # print("y_pred: ", y_pred)
    temporalList = metricExamine(metric, y_test, y_pred)

    if "rocauc" in metric:
        roc_curve_processor(clf, X_test, y_test, "holdout")
        metric.remove("rocauc")

    k = 0
    for i in metric:
        stringResultList.append(i + " : " + str(temporalList[k]))
        k += 1
    # print("Accuracy Score:", metrics.accuracy_score(y_test, y_pred))

    return stringResultList


def printResultWithoutSplit(clf):
    print("Without splitting..?")


def kmeansClustering():
    pass


def supervisedHandler(dataSet, jsonDict, metric):
    header = ["Supervised Learning"]
    iteration = 0
    trainRatio = 0
    # Classifier Logistic Regression
    if jsonDict["classifier"] == "logisticRegression":
        solver = jsonDict["solver"]
        cvalue = float(jsonDict["Cvalforlogistic"])
        clf = LogisticRegression(solver=solver, C=cvalue)
    elif jsonDict["classifier"] == "SVM":
        CvalForSVM = jsonDict["CvalForSVM"]
        gammaForSVM = jsonDict["gammaForSVM"]
        kernelForSVM = jsonDict["kernelForSVM"]
        clf = SVC(
            kernel=kernelForSVM,
            C=float(CvalForSVM),
            gamma=gammaForSVM,
            probability=True,
        )

    elif jsonDict["classifier"] == "KNN":
        nForKNN = jsonDict["nForKNN"]
        disForKNN = jsonDict["disForKNN"]

        if (disForKNN == "1") or (disForKNN == "2"):
            pval = int(disForKNN)

        else:
            pval = int(jsonDict["pvalForMinko"])

        clf = KNeighborsClassifier(n_neighbors=int(nForKNN), p=pval)

    elif jsonDict["classifier"] == "decisionTree":
        criterion = jsonDict["critForTree"]
        maxFeature = jsonDict["maxFeatForTree"]
        maxDepth = jsonDict["maxDepthForTree"]

        # four cases about maxfeatures and maxdepth from json file

        if maxFeature != "None" and maxDepth != "":
            clf = DecisionTreeClassifier(
                criterion=criterion,
                max_features=maxFeature,
                max_depth=int(maxDepth),
            )
        elif maxFeature != "None" and maxDepth == "":
            clf = DecisionTreeClassifier(criterion=criterion, max_features=maxFeature)
        elif maxFeature == "None" and maxDepth != "":
            clf = DecisionTreeClassifier(criterion=criterion, max_depth=int(maxDepth))
        elif maxFeature == "None" and maxDepth == "":
            clf = DecisionTreeClassifier(criterion=criterion)

    if "dataSplit" in jsonDict:

        # holdout
        if jsonDict["splitMethod"] == "holdout":
            trainRatio = 66
            X_train, X_test, y_train, y_test = dataSplitter(trainRatio, dataSet)
            stringResultList = holdoutHandler(
                clf, trainRatio, X_train, X_test, y_train, y_test, metric
            )
        # kfold
        elif jsonDict["splitMethod"] == "kfold":
            stringResultList = kfoldHandler(
                clf,
                int(jsonDict["kfoldValue"]),
                dataSet,
                metric,
                jsonDict["poslabelVal"],
            )

        # Radom Sampling
        else:

            trainRatio = int(jsonDict["range"])
            numOfIter = int(jsonDict["numOfTrial"])
            stringResultList = randomSamplingHandler(
                clf, trainRatio, numOfIter, dataSet, metric, jsonDict["poslabelVal"]
            )

            # print("Average of Accuracy Score: ", meanacc, "(", stdacc, ")")

        return header + stringResultList

    else:
        printResultWithoutSplit()


def unsupervisedHandler(dataSet, jsonDict, metric):
    stringResult = []
    stringResult.append("Unsupervised Learning")
    stringResult.append("Output from downstrem ML")
    stringResult.append("------------------------------------")

    y_true = dataSet[:, -1:]
    y_pred = []
    X = dataSet[:, :-1]

    # Parameter Handling
    if jsonDict["unsupervisedTaskType"] == "K-Means":
        cluster = jsonDict["n_clustersForKmean"]
        ninit = jsonDict["n_init"]
        init = jsonDict["init"]
        clustering = KMeans(
            n_clusters=int(cluster), init=init, n_init=int(ninit), random_state=0
        ).fit(X)

        y_pred = clustering.labels_

        stringResult.append("Method: " + "K-Means")

    elif jsonDict["unsupervisedTaskType"] == "DBSCAN":
        eps = jsonDict["eps"]
        minSamples = jsonDict["min_samples"]
        clustering = DBSCAN(eps=float(eps), min_samples=int(minSamples)).fit(X)
        y_pred = clustering.labels_

    else:  # Agglomerative
        cluster = jsonDict["n_clusters"]
        linkage = jsonDict["linkage"]
        clustering = AgglomerativeClustering(
            n_clusters=int(cluster), linkage=linkage
        ).fit(X)
        y_pred = clustering.labels_
    # Metric Handling
    stringResult.append("------------------------------------")
    stringResult.append("Metric")

    if "RandIndex" in metric:

        score = randIndex(y_true.ravel(), y_pred)
        stringResult.append("Rand Score: " + str(score))

    if "NMI" in metric:

        score = NMI(y_true.ravel(), y_pred)
        stringResult.append("NMI Score: " + str(score))
    if "Silhouette" in metric:
        score = silhouetteScore(X, y_pred)

        stringResult.append("Silhouette Score: " + str(score))

    txtAppender(stringResult)


# Create txt file to store results
txtCreater()
# txtAppender("Output from downstrem ML")
# txtAppender("------------------------")
# Convert JSON input to Dictionary
jsonDict = jsonToDict(sys.argv[1])

# Read numpy array dataset
dataSet = readData()

# In the case where a user chose "All" option
if "all" in jsonDict["metric"]:
    jsonDict["metric"].remove("all")
    # print(jsonDict["metric"])

# Execute supervised machine learning algos
if jsonDict["learningType"] == "supervised":
    txtAppender(supervisedHandler(dataSet, jsonDict, jsonDict["metric"]))
else:
    unsupervisedHandler(dataSet, jsonDict, jsonDict["metric"])
