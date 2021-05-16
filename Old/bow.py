import pandas as pd
import numpy as np
import utilities
from statistics import mean
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import accuracy_score,f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

X_train,y_train=utilities.get_Xy_train(utilities.load_dataset_numpy('train.csv'))
X_train,y_train=utilities.rmv_float_nan(X_train,y_train,2,3)
X_test=utilities.get_X_test(utilities.load_dataset_numpy('test.csv'))
utilities.subs_test_float_nan(X_test,0,1)


# Majority classifier
y_pred=[0]*len(X_train)
acc,f1=utilities.get_acc_f1(y_train,y_pred)
print(acc,f1)
y_submit=[0]*len(X_test)
#utilities.output_submission(y_submit,'majority')


# BOW 
X_train_string_concat=(X_train[:,2:3]+' '+X_train[:,3:4]).squeeze()
X_test_string_concat=(X_test[:,0]+' '+X_test[:,1]).squeeze()

#LogisticRegression
pipe_logreg_bow = Pipeline([
    ('countvectorizer',CountVectorizer()),
    ('logreg', LogisticRegression(max_iter=2500))])
stats_logreg=cross_validate(pipe_logreg_bow,X_train_string_concat,y_train,scoring=['f1','accuracy'],n_jobs=5,verbose=2)
acc=mean(stats_logreg['test_accuracy'])
f1=mean(stats_logreg['test_f1'])
print("Logreg Bow")
print("Acc ",acc,"F1 ",f1)
pipe_logreg_bow.fit(X_train_string_concat,y_train)
y_submit=pipe_logreg_bow.predict(X_test_string_concat)
utilities.output_submission(y_submit,'logreg_bow')

#LinearSVM
pipe_linearsvm_bow = Pipeline([
    ('countvectorizer',CountVectorizer()),
    ('linearsvm', LinearSVC(max_iter=2500))])
stats_svm=cross_validate(pipe_linearsvm_bow,X_train_string_concat,y_train,scoring=['f1','accuracy'],n_jobs=5,verbose=2)
acc=mean(stats_svm['test_accuracy'])
f1=mean(stats_svm['test_f1'])
print("LinearSVM Bow")
print("Acc ",acc,"F1 ",f1)
pipe_linearsvm_bow.fit(X_train_string_concat,y_train)
y_submit=pipe_linearsvm_bow.predict(X_test_string_concat)
utilities.output_submission(y_submit,'linearsvm_bow')