import utils
from utils import QuoraDataDeep,train_deep
from transformers import RobertaForSequenceClassification
from torch.optim import Adam
import torch
import gc

torch.cuda.empty_cache()
gc.collect()
X_dataset,y_dataset=utils.get_Xy_train(utils.load_dataset_numpy('train.csv'))
X_dataset,y_dataset=utils.rmv_float_and_nan(X_dataset,y_dataset)
X_train,y_train,X_val,y_val,X_test,y_test=utils.split_train_dev_test(X_dataset,y_dataset)
quora_dataset=QuoraDataDeep([X_train,X_val,X_test],[y_train,y_val,y_test])
train_loader,val_loader,test_loader=quora_dataset.get_datasets_loaders()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
model.to(device)
optimizer = Adam(model.parameters(), lr=2e-5)
epochs=3
train_deep(model,train_loader,val_loader,optimizer,epochs,device)

