import utils_general
import utils_deep
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
import itertools
import random
from utils_deep import extract_new_examples_idxs_blstm

initial_percentages = [0.02]
increasing_percentages = [0.01, 0.02]
stoppage_percentage=0.25
methods = ["random", "active"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = pd.read_csv("train.csv").to_numpy()
dataset_length = len(dataset)
ratios = [0.7, 0.15, 0.15]
vocab_text = utils_general.Vocab(
    "train.csv", False, range=[0, int(ratios[0] * dataset_length)]
)
vocab_label = utils_general.Vocab("train.csv", True)
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

blstm_log = open("output_blstm_active.txt", "a+")

for init_perc, incr_perc, method in itertools.product(
    initial_percentages, increasing_percentages, methods
):

    blstm_log.write(
        "BLSTM Initial Percentage {} Increasing Percentage {} Method {}\n".format(
            init_perc, incr_perc, method
        )
    )

    train_dataset_current = utils_deep.NLPDataset(
        vocab_text,
        vocab_label,
        "train.csv",
        [0, int(init_perc * ratios[0] * dataset_length)],
    )
    train_current_loader = DataLoader(
        dataset=train_dataset_current,
        batch_size=32,
        shuffle=False,
        collate_fn=utils_deep.pad_collate_fn,
    )
    train_dataset_pooling = utils_deep.NLPDataset(
        vocab_text,
        vocab_label,
        "train.csv",
        [int(init_perc * ratios[0] * dataset_length), int(ratios[0] * dataset_length)],
    )
    train_pooling_loader = DataLoader(
        dataset=train_dataset_pooling,
        batch_size=32,
        shuffle=False,
        collate_fn=utils_deep.pad_collate_fn,
    )

    increasing_number = int(incr_perc * dataset_length)

    while True:

        embedding_matrix = utils_deep.generate_embedding_matrix(
            "sst_glove_6b_300d.txt", vocab_text
        )
        model = utils_deep.BLSTM(embedding_matrix)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        epochs = 5
        optimizer = Adam(model.get_params(), 0.002)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        for epoch in range(epochs):
            utils_deep.train(model, train_current_loader, optimizer, criterion, device)
            scheduler.step()

        metrics_val = utils_deep.evaluate(model, valid_dataloader, criterion, device)
        metrics_test = utils_deep.evaluate(model, test_dataloader, criterion, device)

        blstm_log.write(
            "Train Examples {}, Pooling Examples {}, Val Acc {}, Test Acc {}\n".format(
                len(train_current_loader.dataset),
                len(train_pooling_loader.dataset),
                metrics_val["acc"],
                metrics_test["acc"],
            )
        )
        blstm_log.flush()

        if int(stoppage_percentage*dataset_length) >= len(train_current_loader.dataset):
            break

        new_indexes = extract_new_examples_idxs_blstm(
            model, train_pooling_loader, increasing_number, device, method
        )
        train_current_instances = np.array(train_dataset_current.instances)
        train_pooling_instances = np.array(train_dataset_pooling.instances)
        train_current_instances = np.concatenate(
            (train_current_instances, train_pooling_instances[new_indexes])
        )
        train_pooling_instances = np.delete(
            train_pooling_instances, new_indexes, axis=0
        )

        train_dataset_current = utils_deep.NLPDataset(
            vocab_text, vocab_label, instances=train_current_instances.tolist()
        )
        train_current_loader = DataLoader(
            dataset=train_dataset_current,
            batch_size=32,
            shuffle=False,
            collate_fn=utils_deep.pad_collate_fn,
        )
        train_dataset_pooling = utils_deep.NLPDataset(
            vocab_text, vocab_label, instances=train_pooling_instances.tolist()
        )
        train_pooling_loader = DataLoader(
            dataset=train_dataset_pooling,
            batch_size=32,
            shuffle=False,
            collate_fn=utils_deep.pad_collate_fn,
        )
blstm_log.close()
