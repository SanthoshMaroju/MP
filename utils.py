from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression
import random
import pandas as pd
from sklearn.model_selection import train_test_split

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 2  
    n_features = 4 # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def load_data() -> Dataset:
    
    df = pd.read_csv("D:\Major_Project\MP\IoT_Modbus.csv")
    
    
    df_x = df.drop( ['label','date','time','type'] , axis="columns").values
    df_y = df['label'].values

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.30)


    # df_test = pd.read_csv("D:/Major_Project/MP/Train_Test_IoT_Modbus.csv")
    # df = pd.read_csv("D:/Major_Project/MP/Train_Test_IoT_Modbus.csv")
    # df_test['rand_num'] = [random.random() for _ in range(len(df_test))]
    # df_test = df_test.sort_values(by='rand_num')
    # df_test = df_test.drop(['rand_num'], axis = "columns")
    # df_test = pd.concat([df,df_test])
    # print(len(df), len(df_test))

    # x_test = df_test.drop( ['ts','date','time','type','label'], axis="columns").values
    # y_test = df_test['label'].values

    return (x_train, y_train), (x_test, y_test)


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )
