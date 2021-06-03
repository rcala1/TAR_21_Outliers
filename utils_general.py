import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
from dataclasses import dataclass
import torch


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
        print()
        dataset_new += [dataset[i][0] + " " + dataset[i][1]]
    return dataset_new


def concatenate_sentences_arrays(dataset):
    dataset_new = []
    for i in range(len(dataset)):
        print()
        dataset_new += [list(dataset[i][0]) + list(dataset[i][1])]
    return dataset_new


def most_frequent(List):
    return max(set(List), key=List.count)


@dataclass
class Instance:
    text: list
    label: str


class Vocab:
    def __init__(self, file, label, max_size=-1, min_freq=0, range=None):

        dataset = pd.read_csv(file).to_numpy()
        self.frequencies = {}
        if range is not None:
            dataset = dataset[range[0] : range[1]]

        for line in dataset:
            if label == True:
                splitted = str(line[5]).split()
            else:
                splitted = str(line[3]).split() + str(line[4]).split()
            for token in splitted:
                value = self.frequencies.get(token)
                if value is None:
                    self.frequencies[token] = 1
                else:
                    self.frequencies[token] += 1

        sorted_freq = sorted(
            self.frequencies.items(), key=lambda item: item[1], reverse=True
        )

        self.itos = {}
        self.stoi = {}
        index = -1

        if label == False:
            self.itos = {0: "<PAD>", 1: "<UNK>"}
            self.stoi = {"<PAD>": 0, "<UNK>": 1}
            index = 1

        if max_size != -1:
            sorted_freq = sorted_freq[:max_size]

        for token, value in sorted_freq:
            if value < min_freq:
                break
            index += 1
            self.itos[index] = token
            self.stoi[token] = index

    def encode(self, text):
        if isinstance(text, str):
            return self.encode_string(text)
        else:
            return self.encode_list(text)

    def decode(self, text):
        if isinstance(text, str):
            self.encode_string(text)
        else:
            self.encode_list(text)

    def encode_list(self, text: list):
        encoded_text = []
        for token in text:
            if self.stoi.get(token) is None:
                encoded_text += [1]
            else:
                encoded_text += [self.stoi[token]]
        return torch.IntTensor(encoded_text)

    def encode_string(self, text: str):
        if self.stoi.get(text) is None:
            return 1
        return torch.IntTensor([self.stoi[text]])

    def decode_string(self, encoded: list):
        decoded_text = []
        for index in encoded:
            decoded_text += [self.itos[index]]
        return torch.IntTensor(decoded_text)

    def decode_list(self, encoded: int):
        return torch.IntTensor(self.itos[encoded])
