"""
This module contains the main function.
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from analysis import NaiveBayes, RandomForestClassifier_CV, \
    logisticRegression, DecisionTC, KNN_CV
from preprocessing import preprocessing
from visualization import BoxPlot, correlationMatrix


# ----------------- MISSING VALUE-----------------------------------#
# print(df.isna().sum())


def visualization(df):
    BoxPlot(df)
    correlationMatrix(df)


def comparison(models, scores):
    """

    Parameters
    ----------
    models:classification model analyzed,
    scores: It is a list that memorizes [f1score, recall, precision] for each model in the sequence obtained using models listed in “models.”


    Returns
    -------
    None
    """

    data = {
        'model': models,
        'f1-score': scores[:, 0],
        'Precision': scores[:, 1],
        'Recall': scores[:, 2]
    }

    df = pd.DataFrame(data=data, index=models)
    print(df)

    ax = df.plot.scatter('model', 'f1-score', label='f1-score', s=100)
    df.plot.scatter('model', 'Precision', label='Precision', color='pink', ax=ax, s= 80)
    df.plot.scatter('model', 'Recall', label='Recall', color='orange', ax=ax)
    plt.ylabel('')
    plt.title('Comparison')
    plt.show()

    fullpath = "Results/F1-score.csv"
    df.to_csv(fullpath)
    return


def classification(X_train, X_test, y_train, y_test):
    """
    This function execute various learning models used for the classification task.
    Then take the results of each algorithm and call comparison().

    Parameters
    ----------
    X_train: non-target train,
    X_test: non-target for prediction test,
    y_train: target for prediction train,
    y_test: target for prediction test

    Returns
    -------

    """
    models = ['Knn', 'Naive Bayes', 'Random Forest', 'Decision Tree', 'Logistic Regression']
    scores = []

    # scores['f1-score', 'Precision', 'Recall']
    scores.append(KNN_CV(X_train, X_test, y_train, y_test))
    scores.append(NaiveBayes(X_train, X_test, y_train, y_test))
    scores.append(RandomForestClassifier_CV(X_train, X_test, y_train, y_test))
    scores.append(DecisionTC(X_train, X_test, y_train, y_test))
    scores.append(logisticRegression(X_train, X_test, y_train, y_test))

    scores = np.array(scores)
    scores = np.around(scores, 4)

    shape = (len(models), 3)
    scores_matrix = scores.reshape(shape)

    comparison(models, scores_matrix)


def feature_Importance(df):
    # Filtering df for only good quality
    df_temp = df[df['goodquality'] == 1]
    print(df_temp.describe())
    # Filtering df for only medium quality
    df_temp2 = df[df['goodquality'] == 0]
    print(df_temp2.describe())


if __name__ == "__main__":
    # db = "../Dataset/wineQualityReds.csv"
    db = "../Dataset/whiteWine.csv"
    df, X_train, X_test, y_train, y_test, X = preprocessing(db)

    answer = 'n'
    # answer = input('Do you want visualize your data? Type Y = yes or any other key for = no')
    if (answer == 'Y'):
        visualization(df)
    # feature_Importance(df)

    classification(X_train, X_test, y_train, y_test)
