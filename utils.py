import pandas as pd
import numpy as np
from scipy.sparse import data
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
import torch
import re
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer
import time
from tqdm import tqdm

def load_dataset_numpy(path):
    dataset=pd.read_csv(path)
    dataset_numpy=dataset.to_numpy()
    return dataset_numpy

def get_Xy_train(train_dataset):
    X=train_dataset[:,3:5]
    y=train_dataset[:,5].astype(int)
    return X,y

def rmv_float_and_nan(X,y):
    idxs_del=[]
    for i in range(len(X)):
        try:
            float(X[:,0][i])
            idxs_del+=[i]
            continue
        except ValueError:
            print
            
        try:
            float(X[:,1][i])
            idxs_del+=[i]
            continue
        except ValueError:
            print
    y=np.delete(y,idxs_del,axis=0)
    X=np.delete(X,idxs_del,axis=0) 
    return X,y

def split_train_dev_test(X,y,ratios=[0.7,0.15,0.15]):
    X_train,X_val_test,y_train,y_val_test=train_test_split(X,y,test_size=ratios[1]+ratios[2])
    X_val,X_test,y_val,y_test=train_test_split(X_val_test,y_val_test,test_size=ratios[2]/(ratios[1]+ratios[2]))
    return X_train,y_train,X_val,y_val,X_test,y_test

def remove_punctuation(dataset):
    dataset_new=[]
    for i in range(len(dataset)):
        dataset_new+=[[re.sub(r'[^\w\s]','',dataset[i][0]), \
            re.sub(r'[^\w\s]','',dataset[i][1])]]
    return dataset_new

def split_sentences_in_tokens(dataset):
    dataset_new=[]
    for i in range(len(dataset)):
        dataset_new+=[[dataset[i][0].split() \
            ,dataset[i][1].split()]]
    return dataset_new

class Vocab:

    def __init__(self,dataset,max_size=-1,min_freq=0):

        self.frequencies={}
        for i in range(len(dataset)):
            combined_tokens=dataset[i][0]+dataset[i][1]
            for token in combined_tokens:
                value=self.frequencies.get(token)
                if value is None:
                    self.frequencies[token]=1
                else:
                    self.frequencies[token]+=1

        sorted_freq=sorted(self.frequencies.items(),key=lambda item: item[1],reverse=True)

        self.itos={}
        self.stoi={}
        self.itos={0:"<PAD>",1:"<UNK>"}
        self.stoi={"<PAD>":0,"<UNK>":1}
        index=1

        if max_size!=-1:
            sorted_freq=sorted_freq[:max_size]

        for token, value in sorted_freq:
            if value<min_freq:
                break
            index+=1
            self.itos[index]=token
            self.stoi[token]=index

    def encode(self,text):
        if isinstance(text,str):
            return self.encode_string(text)
        else:
            return self.encode_list(text)

    def decode(self,text):
        if isinstance(text,str):
            self.encode_string(text)
        else:
            self.encode_list(text)

    def encode_list(self,text: list):
        encoded_text=[]
        for token in text:
            if self.stoi.get(token) is None:
                encoded_text+=[1]
            else:
                encoded_text+=[self.stoi[token]]
        return encoded_text

    def encode_string(self,text: str):
        if self.stoi.get(text) is None:
            return 1
        return [self.stoi[text]]

    def decode_string(self,encoded: list):
        decoded_text=[]
        for index in encoded:
            decoded_text+=[self.itos[index]]
        return decoded_text

    def decode_list(self,encoded: int):
        return self.itos[encoded]

def encode_dataset(dataset,vocab):
    dataset_new=[]
    for i in range(len(dataset)):
        dataset_new+=[[vocab.encode(dataset[i][0]) \
            ,vocab.encode(dataset[i][1])]]
    return dataset_new

def concatenate_sentences(dataset):
    dataset_new=[]
    for i in range(len(dataset)):
        dataset_new+=[list(dataset[i][0])+list(dataset[i][1])]
    return dataset_new

def get_largest_length(dataset):
    length=0
    for i in range(len(dataset)):
        if len(dataset[i])>length:
            length=len(dataset[i])
    return length

def pad_dataset(dataset,length):
    dataset_new=[]    
    for i in range(len(dataset)):
        dataset_new+=[np.array(dataset[i].copy())]
        dataset_new[i].resize(length)
    return dataset_new

def load_glove_dict(file_path):
    f = open(file_path, "r")
    glove_dict={}
    for line in f.read().splitlines():
        word,embedding=line.split(" ",1)
        glove_dict[word]=np.array([float(val) for val in embedding.split()])
    glove_dict["<UNK>"]=np.random.randn(300)
    return glove_dict

def encode_glove(glove_dict,dataset,avg_glove=False):

    dataset_new=[]
    for i in range(len(dataset)):
        dataset_new+=[[[],[]]]
        for j in range(len(dataset[i][0])):
            vector=glove_dict.get(dataset[i][0][j])
            if vector is not None:
                dataset_new[i][0]+=[vector]
            else:
                dataset_new[i][0]+=[glove_dict["<UNK>"]]
        for j in range(len(dataset[i][1])):
            vector=glove_dict.get(dataset[i][1][j])
            if vector is not None:
                dataset_new[i][1]+=[vector]
            else:
                dataset_new[i][1]+=[glove_dict["<UNK>"]]
    
    if avg_glove:
        for i in range(len(dataset_new)):
            if not dataset_new[i][0]:
                dataset_new[i][0]=np.ones(300)
            else:
                dataset_new[i][0]=np.mean(dataset_new[i][0],axis=0)
            if not dataset_new[i][1]:
                dataset_new[i][1]=np.ones(300)
            else:
                dataset_new[i][1]=np.mean(dataset_new[i][1],axis=0)
    return dataset_new
    
class QuoraDataDeep(Dataset):

    def __init__(self,datasets_x,datasets_y):
        self.tokenizer=RobertaTokenizer.from_pretrained('roberta-base',do_lower_case=True)
        self.datasets_x=datasets_x
        self.datasets_y=datasets_y

    def process_dataset(self,X,y):
        token_ids=[]
        mask_ids=[]
        seg_ids=[]

        for sentences in X:
            sentence1_id=self.tokenizer.encode(sentences[0],add_special_tokens=False)
            sentence2_id=self.tokenizer.encode(sentences[1],add_special_tokens=False)
            sentences_ids = [self.tokenizer.cls_token_id] + sentence1_id + [self.tokenizer.sep_token_id] + sentence2_id + [self.tokenizer.sep_token_id]
            sentence1_len=len(sentence1_id)
            sentence2_len=len(sentence2_id)

            segment_ids = torch.tensor([0] * (sentence1_len + 2) + [1] * (sentence2_len + 1)) 
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

    def get_datasets_loaders(self,shuffle=True,batch_size=32):

        dataloaders=[]

        for (x,y) in zip(self.datasets_x,self.datasets_y):

            dataset=self.process_dataset(x,y)

            data_loader = DataLoader(
                dataset,
                shuffle=shuffle,
                batch_size=batch_size
            )

            dataloaders+=[data_loader]

        return tuple(dataloaders)

def acc_stat(y_pred, y_true):
  acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_true).sum().float() / float(y_true.size(0))
  return acc

def train_deep(model, train_loader, val_loader, optimizer, epochs,device):  

  start=time.time()

  for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    total_train_acc  = 0
    for pair_token_ids, mask_ids, seg_ids, y in tqdm(train_loader):
      optimizer.zero_grad()
      pair_token_ids = pair_token_ids.to(device)
      mask_ids = mask_ids.to(device)
      seg_ids = seg_ids.to(device)
      labels = y.to(device)

      loss, prediction = model(pair_token_ids, 
                             attention_mask=mask_ids, 
                             labels=labels).values()

      acc = acc_stat(prediction, labels)

      loss.backward()
      optimizer.step()
      
      total_train_loss += loss.item()
      total_train_acc  += acc.item()

    train_acc  = total_train_acc/len(train_loader)
    train_loss = total_train_loss/len(train_loader)

    model.eval()
    total_val_acc  = 0
    total_val_loss = 0

    with torch.no_grad():
      for (pair_token_ids, mask_ids, seg_ids, y) in tqdm(val_loader):
        optimizer.zero_grad()
        pair_token_ids = pair_token_ids.to(device)
        mask_ids = mask_ids.to(device)
        seg_ids = seg_ids.to(device)
        labels = y.to(device)
        
        loss, prediction = model(pair_token_ids, 
                             attention_mask=mask_ids, 
                             labels=labels).values()
        
        acc = acc_stat(prediction, labels)

        total_val_loss += loss.item()
        total_val_acc  += acc.item()

    val_acc  = total_val_acc/len(val_loader)
    val_loss = total_val_loss/len(val_loader)
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))