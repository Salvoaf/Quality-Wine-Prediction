"""
This module implements many functions to transform datasets with preprocessing tasks.
"""
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler, SMOTE
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

np.set_printoptions(threshold=sys.maxsize)


def remove_outlier(db):
    """
    This function aims to remove all outliers using the Box Plot Diagram.
    The purpose of this diagram is to identify outliers and discard it from the data series.

    Parameters
    ----------
    db: DataFrame
        the competition DataFrame

    Yields
    ------
    df: Database cleaned of outliers

    """
    q1 = db.quantile(0.25)
    q2 = db.quantile(0.75)
    IQR = q2 - q1
    df = db[~((db < (q1 - 1.5 * IQR)) | (db > (q2 + 1.5 * IQR))).any(axis=1)]
    return df


def preprocessing(db):
    """
    In this function are done the significant tasks of preprocessing step are. Are defined the targets of interest.
    The dataset is divided into a training set and test set(20%), normalized with MinMaxScaler(). The training set is
    balanced with SMOTE(random_state=42) algorithm.

    Other internal functions are dedicated to plotting some helpful representations of the dataset. If the users want
    them, they must write “plt. show()”.

    Parameters
    ----------
    db: DataFrame the competition DataFrame

    Yields
    ------
    df: DataFrame structured according to our needs,
    X_train: non-target train,
    X_test: target for prediction test,
    y_train: non-target for prediction train,
    y_test: target for prediction test,
    X: attributes
    """

    df = pd.read_csv(db)
    df = df.drop_duplicates(keep='first')
    df = remove_outlier(df)

    # Targets of interest.
    df['goodquality'] = [0 if x < 7 else 1 for x in df['quality']]
    X = df.drop(['quality', 'goodquality'], axis=1)
    y = df['goodquality']

    ##----- PLOT (pre balance) ----##
    sns.countplot(x='goodquality', data=df)

    autopct = "%.2f"
    fig, axs = plt.subplots()
    y.value_counts().plot.pie(autopct=autopct, ax=axs)
    axs.set_title("Original")

    ##----- Sampling -----##
    sampling_strategy = "not majority"
    ros = RandomOverSampler(sampling_strategy=sampling_strategy)
    sm = SMOTE(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)

    y_train_plot = pd.DataFrame(y_train, columns=["goodquality"])
    plt.subplots()
    sns.countplot(x='goodquality', data=y_train_plot)
    plt.title('Original y_train')

    X_train, y_train = sm.fit_resample(X_train, y_train)

    fig, axs = plt.subplots()
    y_train.value_counts().plot.pie(autopct=autopct, ax=axs)
    axs.set_title("Over-sampling")

    fig, axs = plt.subplots()
    y_train_plot = pd.DataFrame(y_train, columns=["goodquality"])
    y_test_plot = pd.DataFrame(y_test, columns=["goodquality"])
    plt.subplots()

    sns.countplot(x='goodquality', data=y_train_plot)
    plt.title('Balanced')
    plt.subplots()
    sns.countplot(data=y_test_plot, x='goodquality')
    plt.title('Test')

    # plt.show()
    fig, axs = plt.subplots()
    slices = [20, 80]
    labels = ['Train', 'Test']

    plt.pie(slices, labels=labels, startangle=90, autopct='%1.1f%%')
    plt.title("Test set and Train set")
    plt.tight_layout()

    # plt.show()

    ##---- Scaling ----##
    sc = MinMaxScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    return df, X_train, X_test, y_train, y_test, X
