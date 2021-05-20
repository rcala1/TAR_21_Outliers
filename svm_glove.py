from re import X
import pandas as pd
import numpy as np
import utils
from utils import Vocab, load_glove_dict
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

X_dataset,y_dataset=utils.get_Xy_train(utils.load_dataset_numpy('train.csv'))
X_dataset,y_dataset=utils.rmv_float_and_nan(X_dataset,y_dataset)
X_dataset=utils.remove_punctuation(X_dataset)
X_dataset=utils.split_sentences_in_tokens(X_dataset)
X_train,y_train,X_val,y_val,X_test,y_test=utils.split_train_dev_test(X_dataset,y_dataset)
glove_dict=load_glove_dict("sst_glove_6b_300d.txt")
X_train=utils.encode_glove(glove_dict,X_train,avg_glove=True)
X_val=utils.encode_glove(glove_dict,X_val,avg_glove=True)
X_test=utils.encode_glove(glove_dict,X_test,avg_glove=True)
X_train=utils.concatenate_sentences(X_train)
X_val=utils.concatenate_sentences(X_val)
X_test=utils.concatenate_sentences(X_test)

svm=LinearSVC()
svm.fit(X_train,y_train)
y_pred_val=svm.predict(X_val)
y_pred_test=svm.predict(X_test)
print(accuracy_score(y_val,y_pred_val))
print(accuracy_score(y_test,y_pred_test))