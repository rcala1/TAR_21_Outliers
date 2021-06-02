import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import time
from tqdm import tqdm
import numpy as np
from utils_general import Instance
import pandas as pd

class QuoraDataBert(Dataset):
    def __init__(self, datasets_x, datasets_y):
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
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


def train_deep(model, train_loader, val_loader, test_loader, optimizer, epochs, device):

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
                    attention_mask=mask_ids,
                    labels=labels,
                ).values()

                acc = acc_stat(prediction, labels)

                total_val_loss += loss.item()
                total_val_acc += acc.item()

        val_acc = total_val_acc / len(val_loader)
        val_loss = total_val_loss / len(val_loader)

        total_test_acc = 0
        total_test_loss = 0

        with torch.no_grad():
            for (pair_token_ids, mask_ids, seg_ids, y) in tqdm(test_loader):
                optimizer.zero_grad()
                pair_token_ids = pair_token_ids.to(device)
                mask_ids = mask_ids.to(device)
                seg_ids = seg_ids.to(device)
                labels = y.to(device)

                loss, prediction = model(
                    pair_token_ids,
                    attention_mask=mask_ids,
                    labels=labels,
                ).values()

                acc = acc_stat(prediction, labels)

                total_test_loss += loss.item()
                total_test_acc += acc.item()

        test_acc = total_test_acc / len(test_loader)
        test_loss = total_test_loss / len(test_loader)

        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)

        print(
            f"Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} \
            | test_loss: {test_loss:.4f} val_acc: {test_acc:.4f}"
        )
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

class BLSTM(nn.Module):

  def __init__(self,embedding_matrix,hidden_size=150,
                num_layers=2,dropout=0,linear_dim=[300,150,1]):
    super().__init__()
    self.embedding_matrix=embedding_matrix
    self.rnn=nn.LSTM(300,hidden_size=hidden_size,num_layers=num_layers,
                        dropout=dropout,bidirectional=True)
    self.fcs=[]
    for i in range(len(linear_dim)-1):
        self.fcs+=[nn.Linear(linear_dim[i],linear_dim[i+1],bias=True)]
    self.fcs=nn.ModuleList(self.fcs)

  def get_params(self):

    params=list()

    params.extend(self.embedding_matrix.parameters())
    for i in range(len(self.fcs)):
      params.extend(self.fcs[i].parameters())
    params.extend(self.rnn.parameters())

    return params

  def forward(self, x):

    x=self.embedding_matrix(x)
    x = torch.transpose(x,0,1)
    h,hidden=self.rnn(x,None)

    h=h[-1]
    for fc in self.fcs[:-1]:
      h=fc(h)
      h=torch.relu(h)
    logits=self.fcs[-1](h)
    return logits

def pad_collate_fn(batch, pad_index=0):

    texts, labels = zip(*batch) 
    lengths = torch.tensor([len(text) for text in texts]) 
    texts=pad_sequence(texts,batch_first=True,padding_value=pad_index)

    return texts, labels, lengths

def train(model, data, optimizer, criterion,device):
  model.train()
  for x, y, idxs in tqdm(data):
    x=x.to(device)
    y=torch.Tensor(y).to(device)
    logits = model(x)
    loss = criterion(logits, y.reshape(len(y),1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

def evaluate(model, data, criterion,device):
  model.eval()
  losses=[]
  y_preds=[]
  ys=[]
  with torch.no_grad():
    for x,y,idxs in data:
      x=x.to(device)
      y=torch.Tensor(y).to(device)
      logits = model(x)
      loss = criterion(logits, y.reshape(len(y),1))
      losses+=[loss.cpu().detach().numpy()]
      y_preds+=(logits>0).long().cpu().detach().numpy().tolist()
      ys+=y.cpu().detach().numpy().tolist()
    ys=np.array(ys)
    y_preds=np.array(y_preds).squeeze()
    acc=np.sum(ys==y_preds)/len(ys)
    return {
        "loss":np.mean(losses),
        "acc":acc,
    }

def generate_embedding_matrix(file,vocab,random_normal=False,freeze=False):

    f = open(file, "r")
    embedding_dict={}
    for line in f.read().splitlines():
        word,embedding=line.split(" ",1)
        embedding_dict[word]=[float(val) for val in embedding.split()]
    
    embedding_matrix=torch.randn(len(vocab.stoi),300)
    embedding_matrix[0]=torch.zeros((300,))

    if random_normal:
        return torch.nn.Embedding.from_pretrained(embedding_matrix,padding_idx=0,freeze=freeze)

    sorted_vocab=sorted(vocab.stoi.items(),key=lambda item: item[1])
    
    for idx, token in enumerate(sorted_vocab):
        if embedding_dict.get(token[0]) is not None:
            embedding_matrix[idx]=torch.Tensor(embedding_dict[token[0]])

    return torch.nn.Embedding.from_pretrained(embedding_matrix,padding_idx=0,freeze=freeze)

class NLPDataset(torch.utils.data.Dataset):

    def __init__(self,vocab_text,vocab_label,file,range=None):
        
        self.instances=[]
        self.vocab_text=vocab_text
        self.vocab_label=vocab_label
        dataset = pd.read_csv(file).to_numpy()
        if range is not None:
            dataset=dataset[range[0]:range[1]]
        for line in dataset:
            first,second,label=line[3:]
            self.instances+=[Instance(str(first).split()+str(second).split(),str(label))]

    def __getitem__(self,idx):
        x_encoded=self.vocab_text.encode(self.instances[idx].text)
        y_encoded=self.vocab_label.encode(self.instances[idx].label)
        return x_encoded,y_encoded

    def __len__(self):
        return len(self.instances)