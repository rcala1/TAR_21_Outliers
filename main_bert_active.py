from utils_deep import (
    QuoraDataBert,
    train_deep,
    evaluate_val,
    evaluate_test,
    extract_new_examples_idxs_bert,
)
import utils_general
from transformers import BertForSequenceClassification
from torch.optim import Adam
import torch
import numpy as np
import itertools

X_dataset, y_dataset = utils_general.get_Xy_train(
    utils_general.load_dataset_numpy("train.csv")
)
X_dataset, y_dataset = utils_general.rmv_float_and_nan(X_dataset, y_dataset)
X_train, y_train, X_val, y_val, X_test, y_test = utils_general.split_train_dev_test(
    X_dataset, y_dataset
)

initial_percentages = [0.02]
increasing_percentages = [0.01, 0.02]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
experiment_combinations = itertools.product(initial_percentages, increasing_percentages)

quora_dataset = QuoraDataBert([X_val, X_test], [y_val, y_test])
val_loader, test_loader = quora_dataset.get_datasets_loaders()


bert_log = open("output_bert_active.txt", "a+")

for init_perc, incr_perc in experiment_combinations:

    bert_log.write(
        "BERT Initial Percentage {} Increasing Percentage {}".format(
            init_perc, incr_perc
        )
    )

    X_original_train = np.array(X_train)
    increasing_number = int(len(X_train) * incr_perc)
    X_train_current = np.array(X_train[: int(len(X_train) * init_perc)])
    X_train_pooling = np.array(X_train[int(len(X_train) * init_perc) :])
    y_train_current = np.array(y_train[: int(len(y_train) * init_perc)])
    y_train_pooling = np.array(y_train[int(len(y_train) * init_perc) :])

    while True:
        
        quora_dataset = QuoraDataBert(
            [X_train_current, X_train_pooling], [y_train_current, y_train_pooling]
        )
        train_current_loader, train_pooling_loader = quora_dataset.get_datasets_loaders(
            shuffle=False
        )

        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=2e-5)
        epochs = 3
        train_deep(model, train_current_loader, val_loader, optimizer, epochs, device)
        val_loss, val_acc = evaluate_val(model, val_loader, device)
        test_loss, test_acc = evaluate_test(model, test_loader, device)

        bert_log.write(
            "Train Examples {}, Pooling Examples, Val Acc {}, Test Acc {}".format(
                len(X_train_current), len(X_train_pooling), val_acc, test_acc
            )
        )

        if len(X_train) == len(X_train_current):
            break

        new_indexes = extract_new_examples_idxs_bert(
            model, train_pooling_loader, increasing_number, device
        )
        X_train_current = np.concatenate(
            (X_train_current, X_train_pooling[new_indexes])
        )
        X_train_pooling = np.delete(X_train_pooling, new_indexes, axis=0)
        y_train_current = np.concatenate(
            (y_train_current, y_train_pooling[new_indexes])
        )
        y_train_pooling = np.delete(y_train_pooling, new_indexes, axis=0)
