import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import metrics
from numpy import mean
from numpy import std

# import pandas as pd
import sys

print("output from downstrem ML")
print("------------------------")


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


def kfoldHandler(clf, kval, dataSet):

    X = dataSet[:, :-1]
    y = dataSet[:, -1:].ravel()
    # prepare the cross-validation procedure
    cv = KFold(n_splits=kval, random_state=1, shuffle=True)
    # evaluate model
    scores = cross_val_score(clf, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
    # report performance
    print("Kvalue for Kfold: ", kval)
    print("Accuracy: %.3f (%.3f)" % (mean(scores), std(scores)))


def accAverageCalculator(clf, trainRatio, numOfIter, dataSet):

    listForAcc = []
    listForVar = []
    for i in range(numOfIter):
        X_train, X_test, y_train, y_test = dataSplitter(trainRatio, dataSet)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        variance = explained_variance_score(y_test, y_pred)
        listForAcc.append(acc)
        listForVar.append(variance)

    # return Average of accuracy scores
    # print(listForAcc)
    # print(listForVar)
    stdOfAcc = std(listForAcc)
    meanOfAcc = sum(listForAcc) / len(listForAcc)
    maxOfAcc = max(listForAcc)
    minOfAcc = min(listForAcc)
    AccPlusError = maxOfAcc - meanOfAcc
    AccMinusError = meanOfAcc - minOfAcc
    meanOfVar = sum(listForVar) / len(listForVar)
    maxOfVar = max(listForVar)
    minOfVar = min(listForVar)
    VarPlusError = maxOfVar - meanOfVar
    VarMinusError = meanOfVar - minOfVar

    return (
        meanOfAcc,
        AccPlusError,
        AccMinusError,
        meanOfVar,
        VarPlusError,
        VarMinusError,
        stdOfAcc,
    )


def printResult(clf, trainRatio, X_train, X_test, y_train, y_test):
    # Result show in console
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("------------------------")
    print("Classifier  : ", clf)
    print("Train ratio : ", str(trainRatio) + "%")
    print("Num of Train: ", len(X_train))
    print("Num of Test : ", len(X_test))
    print("------------------------")
    # print("y_test: ", y_test)
    # print("y_pred: ", y_pred)
    print("Accuracy Score:", metrics.accuracy_score(y_test, y_pred))


def printResultWithoutSplit(clf):
    print("Without splitting..?")


def mlOptionHandler(dataSet, jsonDict):

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
                criterion=criterion, max_features=maxFeature, max_depth=int(maxDepth)
            )
        elif maxFeature != "None" and maxDepth == "":
            clf = DecisionTreeClassifier(criterion=criterion, max_features=maxFeature)
        elif maxFeature == "None" and maxDepth != "":
            clf = DecisionTreeClassifier(criterion=criterion, max_depth=int(maxDepth))
        elif maxFeature == "None" and maxDepth == "":
            clf = DecisionTreeClassifier(criterion=criterion)

    if "dataSplit" in jsonDict:

        if jsonDict["splitMethod"] == "holdout":
            trainRatio = 66
            X_train, X_test, y_train, y_test = dataSplitter(trainRatio, dataSet)
            printResult(clf, trainRatio, X_train, X_test, y_train, y_test)
        elif jsonDict["splitMethod"] == "kfold":
            kfoldHandler(clf, int(jsonDict["kfoldValue"]), dataSet)

        else:
            ## Radom Sampling
            trainRatio = int(jsonDict["range"])
            numOfIter = int(jsonDict["numOfTrial"])
            (
                meanOfAcc,
                AccPlusError,
                AccMinusError,
                meanOfVar,
                VarPlusError,
                VarMinusError,
                stdOfAcc,
            ) = accAverageCalculator(clf, trainRatio, numOfIter, dataSet)
            print("------------------------")
            print("Num of Trials: ", numOfIter)
            print(
                "Average of Accuracy Score: ",
                meanOfAcc,
                " (+ ",
                AccPlusError,
                ")",
                " (- ",
                AccMinusError,
                ")",
                "stdOfAccuracy: ",
                stdOfAcc,
            )
            """
            print(
                "Average of Variances     : ",
                meanOfVar,
                " + ",
                VarPlusError,
                " - ",
                VarMinusError,
            )
            """
    else:
        printResultWithoutSplit()


# Convert JSON input to Dictionary
jsonDict = jsonToDict(sys.argv[1])

# Read numpy array dataset
dataSet = readData()

# Execute machine learning algos
mlOptionHandler(dataSet, jsonDict)
