import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,f1_score

def load_dataset_numpy(path):
    dataset=pd.read_csv(path)
    dataset_numpy=dataset.to_numpy()
    return dataset_numpy

def get_Xy_train(train_dataset):
    X=train_dataset[:,1:5]
    y=train_dataset[:,5].astype(int)
    return X,y

def get_X_test(test_dataset):
    return test_dataset[:,1:]

def rmv_float_nan(X,y,idx1,idx2):
    
    idxs_del=[]
    for i in range(len(X)):
        try:
            float(X[:,idx1][i])
            idxs_del+=[i]
            continue
        except ValueError:
            print
            
        try:
            float(X[:,idx2][i])
            idxs_del+=[i]
            continue
        except ValueError:
            print
    y=np.delete(y,idxs_del,axis=0)
    X=np.delete(X,idxs_del,axis=0) 
    return X,y

def subs_test_float_nan(X,idx1,idx2):
    
    for i in range(len(X)):
        try:
            float(X[:,idx1][i])
            X[i,idx1]=''
            continue
        except ValueError:
            print
            
        try:
            float(X[:,idx2][i])
            X[i,idx2]=''
            continue
        except ValueError:
            print

def get_acc_f1(y_true,y_pred):
    acc=accuracy_score(y_true,y_pred)
    f1=f1_score(y_true,y_pred)
    return acc,f1

def output_submission(y_submit,prefix):
    submission=pd.read_csv('sample_submission.csv')
    submission['is_duplicate']=y_submit
    submission.to_csv(prefix+'_submission.csv',index=False)
    return