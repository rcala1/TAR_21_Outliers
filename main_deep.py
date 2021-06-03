# BERT

import utils_general
from utils_deep import QuoraDataBert, train_deep, evaluate_test
from transformers import BertForSequenceClassification
from torch.optim import Adam
import torch

X_dataset, y_dataset = utils_general.get_Xy_train(
    utils_general.load_dataset_numpy("train.csv")
)
X_dataset, y_dataset = utils_general.rmv_float_and_nan(X_dataset, y_dataset)
X_train, y_train, X_val, y_val, X_test, y_test = utils_general.split_train_dev_test(
    X_dataset, y_dataset
)
quora_dataset = QuoraDataBert([X_train, X_val, X_test], [y_train, y_val, y_test])
train_loader, val_loader, test_loader = quora_dataset.get_datasets_loaders()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = model.to(device)
optimizer = Adam(model.parameters(), lr=2e-5)
epochs = 3
train_deep(model, train_loader, val_loader, test_loader, optimizer, epochs, device)
evaluate_test(model, test_loader, device)


# Active LR BERT - testing

from utils_deep import (
    QuoraDataBert,
    evaluate_val,
    evaluate_test,
    extract_new_examples_idxs,
)
from transformers import BertForSequenceClassification
from torch.optim import Adam
import torch
from tqdm import tqdm
import numpy as np

X_dataset, y_dataset = utils_general.get_Xy_train(
    utils_general.load_dataset_numpy("/content/gdrive/MyDrive/Outliers/train.csv")
)
X_dataset, y_dataset = utils_general.rmv_float_and_nan(X_dataset, y_dataset)
X_train, y_train, X_val, y_val, X_test, y_test = utils_general.split_train_dev_test(
    X_dataset, y_dataset
)

initial_percent = 0.05
increasing_percentage = 0.05
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X_original_train = np.array(X_train)
X_train_current = np.array(X_train[: int(len(X_train) * initial_percent)])
X_train_pooling = np.array(X_train[int(len(X_train) * initial_percent) :])
y_train_current = np.array(y_train[: int(len(y_train) * initial_percent)])
y_train_pooling = np.array(y_train[int(len(y_train) * initial_percent) :])

quora_dataset = QuoraDataBert([X_val, X_test], [y_val, y_test])
val_loader, test_loader = quora_dataset.get_datasets_loaders()
val_accs = [0, 0]

while True:
    quora_dataset = QuoraDataBert(
        [X_train_current, X_train_pooling], [y_train_current, y_train_pooling]
    )
    train_current_loader, train_pooling_loader = quora_dataset.get_datasets_loaders()

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=2e-5)
    epochs = 2
    # train_deep(model, train_current_loader, val_loader, optimizer, epochs, device)
    new_indexes = extract_new_examples_idxs(
        model, train_pooling_loader, increasing_percentage, device
    )
    X_train_current = np.concatenate((X_train_current, X_train_pooling[new_indexes]))
    X_train_pooling = np.delete(X_train_pooling, new_indexes, axis=0)
    y_train_current = np.concatenate((y_train_current, y_train_pooling[new_indexes]))
    y_train_pooling = np.delete(y_train_pooling, new_indexes, axis=0)
    val_loss, val_acc = evaluate_val(model, val_loader, device)
    if val_accs[-1] < val_accs[-2] and val_acc < val_accs[-1]:
        break
    else:
        val_accs += [val_acc]
evaluate_test(model, test_loader, device)


# BLSTM

import utils_general
import utils_classic
import utils_deep
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import pandas as pd
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = pd.read_csv("train.csv").to_numpy()
dataset_length = len(dataset)
ratios = [0.7, 0.15, 0.15]
vocab_text = utils_general.Vocab(
    "train.csv", False, range=[0, int(ratios[0] * dataset_length)]
)
vocab_label = utils_general.Vocab("train.csv", True)
embedding_matrix = utils_deep.generate_embedding_matrix(
    "sst_glove_6b_300d.txt", vocab_text
)
train_dataset = utils_deep.NLPDataset(
    vocab_text, vocab_label, "train.csv", [0, int(ratios[0] * dataset_length)]
)
val_dataset = utils_deep.NLPDataset(
    vocab_text,
    vocab_label,
    "train.csv",
    [int(ratios[0] * dataset_length), int((ratios[0] + ratios[1]) * dataset_length)],
)
test_dataset = utils_deep.NLPDataset(
    vocab_text,
    vocab_label,
    "train.csv",
    [int((ratios[0] + ratios[1]) * dataset_length), dataset_length],
)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=utils_deep.pad_collate_fn,
)
valid_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=utils_deep.pad_collate_fn,
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=utils_deep.pad_collate_fn,
)

model = utils_deep.BLSTM(embedding_matrix)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.get_params(), 0.002)
epochs = 5
scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

for epoch in range(epochs):
    utils_deep.train(model, train_dataloader, optimizer, criterion, device)
    metrics = utils_deep.evaluate(model, valid_dataloader, criterion, device)
    print("Valid Loss {}, Acc {}".format(metrics["loss"], metrics["acc"]))
    scheduler.step()
    print(scheduler.get_lr())
metrics = utils_deep.evaluate(model, test_dataloader, criterion, device)
print("Test Loss {}, Acc {}".format(metrics["loss"], metrics["acc"]))
