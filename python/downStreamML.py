import json
import os
import numpy as np
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
from sklearn import metrics
from sklearn import preprocessing
from numpy import mean
from numpy import std
from txtHandler import *
from metric import *


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
    stringResultList.append("Kvalue for Kfold: " + str(kval))
    X = dataSet[:, :-1]
    y = dataSet[:, -1:].ravel()
    # y = preprocessing.label_binarize(y, classes=[0, 1])
    # print(y[:500])
    # prepare the cross-validation procedure

    cv = KFold(n_splits=kval, random_state=1, shuffle=True)

    # evaluate model
    if posVal == "":
        posVal = 1
    temporalResultList = kfoldMetric(cv, clf, metric, X, y, posVal)
    stringResultList = stringResultList + temporalResultList
    accuracy = cross_val_score(clf, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
    print(stringResultList)
    # report performance

    # print("Accuracy: %.3f (%.3f)" % (mean(accuracy), std(accuracy)))
    # print("precision:", precision)
    # return mean(accuracy), std(accuracy)
    return mean(accuracy), std(accuracy), stringResultList


def classGetter(y):

    return list(set(y.ravel()))


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
            """
            clfName = clf.__class__.__name__
            y_score = clf.fit(X_train, y_train).decision_function(X_test)
            # probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
            tprs = []
            aucs = []

            if posVal == "":
                posVal = 1
            for j in range(n_classes):

                fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=int(posVal))

                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                plt.plot(
                    fpr,
                    tpr,
                    lw=1,
                    alpha=0.3,
                    label="ROC fold %d (AUC = %0.2f)" % (i, roc_auc),
                )
                plt.plot(
                    [0, 1],
                    [0, 1],
                    linestyle="--",
                    lw=2,
                    color="r",
                    label="Chance",
                    alpha=0.8,
                )
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            plt.plot(
                mean_fpr,
                mean_tpr,
                color="b",
                label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
                lw=2,
                alpha=0.8,
            )

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(
                mean_fpr,
                tprs_lower,
                tprs_upper,
                color="grey",
                alpha=0.2,
                label=r"$\pm$ 1 std. dev.",
            )

            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.xlabel("False Positive Rate", fontsize=18)
            plt.ylabel("True Positive Rate", fontsize=18)
            plt.title("Cross-Validation ROC of " + clfName, fontsize=18)
            plt.legend(loc="lower right", prop={"size": 15})

            path = Path(__file__).parent.absolute()
            strFile = str(path) + "/results" + "/rocCurve.png"
            plt.savefig(strFile)
            plt.show()
            """
        # print(y_test)
        # print("y_pred", y_pred)
    temporalList = metricExamine(metric, y_test, y_pred)

    # Accumulate each metric values to dictionary
    for j in range(len(temporalList)):
        temporalDict[metric[j]].append(temporalList[j])

    # Compute ROC curve and ROC area for each class
    # probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
    ##fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    """
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    """

    # print(temporalDict)
    values = temporalDict.values()
    for k, p in zip(values, range(len(metric))):

        temporalDict[metric[p]] = mean(k), std(k)
        # print(type(temporalDict[metric[p]]))

    # print("after:", temporalDict)

    for k, v in temporalDict.items():
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
        clf = LogisticRegression(solver=solver, C=cvalue, random_state=0)
    elif jsonDict["classifier"] == "SVM":
        CvalForSVM = jsonDict["CvalForSVM"]
        gammaForSVM = jsonDict["gammaForSVM"]
        kernelForSVM = jsonDict["kernelForSVM"]
        clf = SVC(kernel=kernelForSVM, C=float(CvalForSVM), gamma=gammaForSVM)

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
            mean, std, stringResultList = kfoldHandler(
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

# Execute supervised machine learning algos
if jsonDict["learningType"] == "supervised":
    txtAppender(supervisedHandler(dataSet, jsonDict, jsonDict["metric"]))
else:
    unsupervisedHandler(dataSet, jsonDict, jsonDict["metric"])
