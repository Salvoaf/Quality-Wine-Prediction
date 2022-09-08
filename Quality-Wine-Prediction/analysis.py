"""
This module contains all algorithms used for the classification task.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=sys.maxsize)

scor = 'f1'


def csv_algo(model, grid, y_test, y_pred):
    """
        This function prints the algorithm’s results in a CSV file.
        
        Parameters
        ----------
        model: classification model analyzed,
        grid: results of GridsearchCV from the model,
        y_test: target for prediction train,
        y_pred: the target used for prediction

        Yields
        ------
        Scores: It is a list that contains the f1 average, precision average, and recall average.
    """
    # print(grid.cv_results_)
    # print(grid.best_params_)

    df = pd.DataFrame(grid.cv_results_)[
        ['params', 'mean_fit_time', 'std_fit_time', 'mean_test_score', 'std_test_score']]

    df = df.round(decimals=6)
    fullpath = "Results_gridsearch/" + str(model) + ".csv"
    df.to_csv(fullpath)

    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    fullpath = "Results_report/" + str(model) + "_best.csv"
    df = df.round(decimals=6)
    df.to_csv(fullpath)

    macro_f1 = report['macro avg']['f1-score']
    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']

    scores = [macro_f1, macro_precision, macro_recall]
    return scores


def KNN_CV(X_train, X_test, y_train, y_test):
    """
        This function is used to visualize the results of the grid searches for the k nearest neighbor algorithm.
        We used GridSearchCV to test Hyper-parameters with cross-validation algorithms kFold.

        In kFold, the data is shuffled once at the start, and then divided into the number of desired splits.

        The only Hyper-parameters are used are the n_neighbors parameters which contains
        the interval from 2 to k neighbors, and they test among all these values which one gives the best results

        Parameters
        ----------
        X_train: non-target train,
        X_test: non-target for prediction test,
        y_train: target for prediction train,
        y_test: target for prediction test

         

        Yields
        ------
        score:  it is the result of the csv_algo(model, grid, y_test, y_pred) function. A list with [
                macro_f1, macro_precision, macro_recall] computed from report().
    """

    kFold = KFold(n_splits=10)  # n_split: Returns the number of splitting iterations in the cross-validator

    n_neighbors = list(range(2, 40))

    knn = KNeighborsClassifier()
    grid_param = {'n_neighbors': n_neighbors}

    cvs = GridSearchCV(knn, grid_param, scoring=scor, n_jobs=-1, cv=kFold, refit=True).fit(X_train, y_train)
    y_prediction = cvs.predict(X_test)

    score = csv_algo(knn, cvs, y_test, y_prediction)

    return score


def RandomForestClassifier_CV(X_train, X_test, y_train, y_test):
    """
    This function is used to visualize the results of the grid searches for the random
    forest algorithm.

    Parameters
    ----------
    X_train: non-target train,
    X_test: non-target for prediction test,
    y_train: target for prediction train,
    y_test: target for prediction test

    Yields
    -------
    score:  it is the result of the csv_algo(model, grid, y_test, y_pred) function. A list with [macro_f1, macro_precision, macro_recall] computed from report()
    """



    # -----------------------RANDOM FOREST CLASSIFIER--------------------------------------#
    grid_param = {
        'n_estimators': [600, 1000],
        'max_depth': [6, 10],
        'criterion': ["entropy"],
        'bootstrap': [False]
    }

    cv_inner = KFold(n_splits=6, shuffle=True, random_state=1)  # StratifiedKFold(n_splits=5, shuffle=True)
    model = RandomForestClassifier(random_state=1)

    search = GridSearchCV(model, grid_param, scoring=scor, n_jobs=-1, cv=cv_inner, refit=True)
    search.fit(X_train, y_train)

    y_prediction = search.predict(X_test)

    score = csv_algo(model, search, y_test, y_prediction)

    # Compute train and test errors VUOI PLOTTARE TRAIN E TEST ERROR PER LE SLIDE

    return score


def NaiveBayes(X_train, X_test, y_train, y_test):
    """
        This function is used to visualize the results of the grid searches for the Naive Bayes algorithm.
        As hyper-parameters, we used var_smoothing.

        It is a stability calculation to widen (or smooth) the curve and therefore account for more samples that are
        further away from the distribution mean.

        Parameters
        ----------
        X_train: non-target train,
        X_test: target for prediction test,
        y_train: target for prediction train,
        y_test: target for prediction test
         

        Yields
        ------
        score:  it is the result of the csv_algo(model, grid, y_test, y_pred) function. A list with [
                macro_f1, macro_precision, macro_recall] computed from report().
    """
    param_grid_nb_Gaussian = {
        'var_smoothing': np.logspace(0, -7, num=10)
    }

    # var_smoothing is a stability calculation to widen (or smooth)
    # the curve and therefore account for more samples that are further away
    # from the distribution mean. In this case, np.logspace returns numbers spaced evenly
    # on a log scale, starts from 0, ends at -9, and generates 100 samples.

    # Creating and training the Complement Naive Bayes Classifier
    param_grid_nb_Multinomial = {'alpha': [0.01, 0.1, 0.5, 1.0, 10.0]}
    param_grid_nb = {}
    model = GaussianNB()
    search = GridSearchCV(model, param_grid=param_grid_nb_Gaussian, scoring=scor, cv=3, n_jobs=-1)
    search.fit(X_train, y_train)

    y_prediction = search.predict(X_test)

    score = csv_algo(model, search, y_test, y_prediction)

    return score


def logisticRegression(X_train, X_test, y_train, y_test):
    """
        This function is used to visualize the results of the grid searches for the Logistic Regression algorithm.
        As hyper-parameter we used C .It’s a penalty term, meant to disincentivize and regulate against Overfitting.

        Parameters
        ----------
        X_train: non-target train,
        X_test: target for prediction test,
        y_train: non-target for prediction train,
        y_test: target for prediction test,
         

        Returns
        ------
        score:  it is the result of the csv_algo(model, grid, y_test, y_pred) function. A list with [
                macro_f1, macro_precision, macro_recall] computed from report().
     """
    param_grid = {'C': [0.001, 100, 1000]}
    model = LogisticRegression()
    search = GridSearchCV(model, param_grid, scoring=scor)
    search.fit(X_train, y_train)

    y_prediction = search.predict(X_test)

    score = csv_algo(model, search, y_test, y_prediction)

    return score


def DecisionTC(X_train, X_test, y_train, y_test):
    """
        This function is used to visualize the results of the grid searches for the Decision Tree algorithm.

        Parameters
        ----------
        X_train: non-target train,
        X_test: non-target for prediction test,
        y_train: target for prediction train,
        y_test: target for prediction test
         

        Yields
        ------
        score:  it is the result of the csv_algo(model, grid, y_test, y_pred) function. A list with [
                macro_f1, macro_precision, macro_recall] computed from report().
    """

    grid_param = {
        'max_depth': [6, 13, 20],
        'criterion': ["gini", "entropy"],
    }
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
    model = DecisionTreeClassifier(random_state=1)
    search = GridSearchCV(model, grid_param, scoring=scor, n_jobs=-1, cv=cv_inner, refit=True)
    search.fit(X_train, y_train)
    y_prediction = search.predict(X_test)

    score = csv_algo(model, search, y_test, y_prediction)

    return score


