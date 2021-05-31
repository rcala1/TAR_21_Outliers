import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re

def load_dataset_numpy(path):
    dataset = pd.read_csv(path)
    dataset_numpy = dataset.to_numpy()
    return dataset_numpy


def get_Xy_train(train_dataset):
    X = train_dataset[:, 3:5]
    y = train_dataset[:, 5].astype(int)
    return X, y


def rmv_float_and_nan(X, y):
    idxs_del = []
    for i in range(len(X)):
        try:
            float(X[:, 0][i])
            idxs_del += [i]
            continue
        except ValueError:
            print

        try:
            float(X[:, 1][i])
            idxs_del += [i]
            continue
        except ValueError:
            print
    y = np.delete(y, idxs_del, axis=0)
    X = np.delete(X, idxs_del, axis=0)
    return X, y


def split_train_dev_test(X, y, ratios=[0.7, 0.15, 0.15]):
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y, test_size=ratios[1] + ratios[2]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=ratios[2] / (ratios[1] + ratios[2])
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def remove_punctuation(dataset):
    dataset_new = []
    for i in range(len(dataset)):
        dataset_new += [
            [
                re.sub(r"[^\w\s]", "", dataset[i][0]),
                re.sub(r"[^\w\s]", "", dataset[i][1]),
            ]
        ]
    return dataset_new


def split_sentences_in_tokens(dataset):
    dataset_new = []
    for i in range(len(dataset)):
        dataset_new += [[dataset[i][0].split(), dataset[i][1].split()]]
    return dataset_new


def concatenate_sentences(dataset):
    dataset_new = []
    for i in range(len(dataset)):
        dataset_new += [list(dataset[i][0]) + list(dataset[i][1])]
    return dataset_new

def most_frequent(List):
    return max(set(List), key = List.count)
