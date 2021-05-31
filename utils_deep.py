import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import time
from tqdm import tqdm


class QuoraDataBert(Dataset):
    def __init__(self, datasets_x, datasets_y):
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base_uncased", do_lower_case=True
        )
        self.datasets_x = datasets_x
        self.datasets_y = datasets_y

    def process_dataset(self, X, y):
        token_ids = []
        mask_ids = []
        seg_ids = []

        for sentences in X:
            sentence1_id = self.tokenizer.encode(sentences[0], add_special_tokens=False)
            sentence2_id = self.tokenizer.encode(sentences[1], add_special_tokens=False)
            sentences_ids = (
                [self.tokenizer.cls_token_id]
                + sentence1_id
                + [self.tokenizer.sep_token_id]
                + sentence2_id
                + [self.tokenizer.sep_token_id]
            )
            sentence1_len = len(sentence1_id)
            sentence2_len = len(sentence2_id)

            segment_ids = torch.tensor(
                [0] * (sentence1_len + 2) + [1] * (sentence2_len + 1)
            )
            attention_mask_ids = torch.tensor([1] * (sentence1_len + sentence2_len + 3))

            token_ids.append(torch.tensor(sentences_ids))
            seg_ids.append(segment_ids)
            mask_ids.append(attention_mask_ids)

        token_ids = pad_sequence(token_ids, batch_first=True)
        mask_ids = pad_sequence(mask_ids, batch_first=True)
        seg_ids = pad_sequence(seg_ids, batch_first=True)
        y = torch.tensor(y)
        dataset = TensorDataset(token_ids, mask_ids, seg_ids, y)
        return dataset

    def get_datasets_loaders(self, shuffle=True, batch_size=32):

        dataloaders = []

        for (x, y) in zip(self.datasets_x, self.datasets_y):

            dataset = self.process_dataset(x, y)

            data_loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

            dataloaders += [data_loader]

        return tuple(dataloaders)


def acc_stat(y_pred, y_true):
    acc = (
        torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_true
    ).sum().float() / float(y_true.size(0))
    return acc


def train_deep(model, train_loader, val_loader, optimizer, epochs, device):

    start = time.time()

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        for pair_token_ids, mask_ids, seg_ids, y in tqdm(train_loader):
            optimizer.zero_grad()
            pair_token_ids = pair_token_ids.to(device)
            mask_ids = mask_ids.to(device)
            seg_ids = seg_ids.to(device)
            labels = y.to(device)

            loss, prediction = model(
                pair_token_ids,
                mask_ids=mask_ids,
                attention_mask=mask_ids,
                labels=labels,
            ).values()

            acc = acc_stat(prediction, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_acc += acc.item()

        train_acc = total_train_acc / len(train_loader)
        train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_acc = 0
        total_val_loss = 0

        with torch.no_grad():
            for (pair_token_ids, mask_ids, seg_ids, y) in tqdm(val_loader):
                optimizer.zero_grad()
                pair_token_ids = pair_token_ids.to(device)
                mask_ids = mask_ids.to(device)
                seg_ids = seg_ids.to(device)
                labels = y.to(device)

                loss, prediction = model(
                    pair_token_ids,
                    mask_ids=mask_ids,
                    attention_mask=mask_ids,
                    labels=labels,
                ).values()

                acc = acc_stat(prediction, labels)

                total_val_loss += loss.item()
                total_val_acc += acc.item()

        val_acc = total_val_acc / len(val_loader)
        val_loss = total_val_loss / len(val_loader)
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)

        print(
            f"Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}"
        )
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
