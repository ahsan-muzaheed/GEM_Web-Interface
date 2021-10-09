from sklearn.metrics import *
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
import matplotlib.patches as patches
import matplotlib.pylab as plt
import numpy as np  # linear algebra
import pandas as pd
from pathlib import Path
from txtHandler import *
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_samples, silhouette_score

# https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python


def accuracyMetric(y_test, y_pred):
    return accuracy_score(y_test, y_pred)


def precisionMetric(y_test, y_pred):
    return precision_score(y_test, y_pred, average="weighted", labels=np.unique(y_pred))


def recallMetric(y_test, y_pred):
    return recall_score(y_test, y_pred, average="weighted", labels=np.unique(y_pred))


def f1ScoreMetric(y_test, y_pred):
    return f1_score(y_test, y_pred, average="weighted", labels=np.unique(y_pred))


def kfoldMetric(cv, clf, metric, X, y, posVal):
    stringResultList = []
    clfName = clf.__class__.__name__
    if "accuracy" in metric:
        accuracy = cross_val_score(clf, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
        stringResultList.append(
            "Accuracy: %.3f (%.3f)" % (mean(accuracy), std(accuracy))
        )
    if "precision" in metric:
        precision_micro = cross_val_score(
            clf, X, y, scoring="precision_micro", cv=cv, n_jobs=-1
        )

        stringResultList.append(
            "Precision_micro: %.3f (%.3f)"
            % (mean(precision_micro), std(precision_micro))
        )
    if "f1_score" in metric:
        f1_score_micro = cross_val_score(
            clf, X, y, scoring="f1_micro", cv=cv, n_jobs=-1
        )
        f1_score_macro = cross_val_score(
            clf, X, y, scoring="f1_macro", cv=cv, n_jobs=-1
        )

        stringResultList.append(
            "f1_score_micro: %.3f (%.3f)" % (mean(f1_score_micro), std(f1_score_micro))
        )
        stringResultList.append(
            "f1_score_macro : %.3f (%.3f)" % (mean(f1_score_macro), std(f1_score_macro))
        )
    if "recall" in metric:
        recall_micro = cross_val_score(
            clf, X, y, scoring="recall_micro", cv=cv, n_jobs=-1
        )
        recall_macro = cross_val_score(
            clf, X, y, scoring="recall_macro", cv=cv, n_jobs=-1
        )
        stringResultList.append(
            "recall_micro: %.3f (%.3f)" % (mean(recall_micro), std(recall_micro))
        )
        stringResultList.append(
            "recall_macro: %.3f (%.3f)" % (mean(recall_macro), std(recall_macro))
        )
    if "rocauc" in metric:
        """
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        n_classes = np.unique(y)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for train, test in cv.split(X, y):

            probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
            for i in range(n_classes):
                fpr[i], tpr[i], t = roc_curve(y[test], probas_[:, 1])
                print(fpr[i], tpr[i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        """
        stringResultList.append("positive Label: ", posVal)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        plt.figure(figsize=(10, 10))
        i = 0
        for train, test in cv.split(X, y):
            probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(
                y[test], probas_[:, 1], pos_label=int(posVal)
            )

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

            i += 1
        plt.plot(
            [0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8
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
    return stringResultList


def metricExamine(metriclist, y_test, y_pred):
    temporalList = []
    if "accuracy" in metriclist:
        temporalList.append(accuracyMetric(y_test, y_pred))

    if "precision" in metriclist:
        temporalList.append(precisionMetric(y_test, y_pred))

    if "recall" in metriclist:
        temporalList.append(recallMetric(y_test, y_pred))

    if "f1_score" in metriclist:
        temporalList.append(f1ScoreMetric(y_test, y_pred))

    # if "rocauc" in metriclist:

    return temporalList


def randIndex(y_true, y_pred):
    return rand_score(y_true, y_pred)


def NMI(y_true, y_pred):
    return adjusted_mutual_info_score(y_true, y_pred)


def silhouetteScore(X, y_pred):
    return silhouette_score(X, y_pred, metric="euclidean")
