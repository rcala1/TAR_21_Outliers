

"""
#BERT
 
import utils_general
from utils_deep import QuoraDataBert, train_deep
from transformers import BertForSequenceClassification
from torch.optim import Adam
import torch
import gc

torch.cuda.empty_cache()
gc.collect()
X_dataset, y_dataset = utils_general.get_Xy_train(utils_general.load_dataset_numpy("train.csv"))
X_dataset, y_dataset = utils_general.rmv_float_and_nan(X_dataset, y_dataset)
X_train, y_train, X_val, y_val, X_test, y_test = utils_general.split_train_dev_test(
    X_dataset, y_dataset
)
quora_dataset = QuoraDataBert([X_train, X_val, X_test], [y_train, y_val, y_test])
train_loader, val_loader, test_loader = quora_dataset.get_datasets_loaders()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)
optimizer = Adam(model.parameters(), lr=2e-5)
epochs = 3
train_deep(model, train_loader, val_loader, test_loader, optimizer, epochs, device)

"""

#BLSTM

import utils_general
import utils_classic
import utils_deep
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import pandas as pd

"""
X_dataset, y_dataset = utils_general.get_Xy_train(utils_general.load_dataset_numpy("train.csv"))
X_dataset, y_dataset = utils_general.rmv_float_and_nan(X_dataset, y_dataset)
X_dataset = utils_general.remove_punctuation(X_dataset)
X_train, y_train, X_val, y_val, X_test, y_test = utils_general.split_train_dev_test(
    X_dataset, y_dataset
)
"""

dataset=pd.read_csv("train.csv").to_numpy()
dataset_length=len(dataset)
ratios=[0.7,0.15,0.15]
vocab_text=utils_general.Vocab("train.csv",False)
vocab_label=utils_general.Vocab("train.csv",True)
embedding_matrix=utils_deep.generate_embedding_matrix('sst_glove_6b_300d.txt',vocab_text)
train_dataset=utils_deep.NLPDataset(vocab_text,vocab_label,"train.csv",[0,int(ratios[0]*dataset_length)])
val_dataset=utils_deep.NLPDataset(vocab_text,vocab_label,"train.csv",[int(ratios[0]*dataset_length),int((ratios[0]+ratios[1])*dataset_length)])
test_dataset=utils_deep.NLPDataset(vocab_text,vocab_label,"train.csv",[int((ratios[0]+ratios[1])*dataset_length),dataset_length])
train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, 
                              shuffle=True, collate_fn=utils_deep.pad_collate_fn)
valid_dataloader = DataLoader(dataset=val_dataset, batch_size=32, 
                              shuffle=True, collate_fn=utils_deep.pad_collate_fn)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, 
                              shuffle=True, collate_fn=utils_deep.pad_collate_fn)

model=utils_deep.BLSTM(embedding_matrix)
criterion=nn.BCEWithLogitsLoss()
optimizer=Adam(model.get_params(),0.005)
epochs=2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    utils_deep.train(model,train_dataloader,optimizer,criterion,device)
    metrics=utils_deep.evaluate(model,valid_dataloader,criterion,device)
    print("Valid Loss {}, Acc {}".format(metrics['loss'],metrics['acc']))
metrics=utils_deep.evaluate(model,test_dataloader,criterion,device)
print("Test Loss {}, Acc {}".format( metrics['loss'],metrics['acc']))