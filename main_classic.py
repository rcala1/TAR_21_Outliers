import utils_general
import utils_classic
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

X_dataset, y_dataset = utils_general.get_Xy_train(utils_general.load_dataset_numpy("train.csv"))
X_dataset, y_dataset = utils_general.rmv_float_and_nan(X_dataset, y_dataset)
X_dataset = utils_general.remove_punctuation(X_dataset)
X_train, y_train, X_val, y_val, X_test, y_test = utils_general.split_train_dev_test(
    X_dataset, y_dataset
)

"""
#glove
glove_dict = utils_classic.load_glove_dict("sst_glove_6b_300d.txt")
X_train = utils_classic.encode_glove(glove_dict, X_train, avg_glove=True)
X_val = utils_classic.encode_glove(glove_dict, X_val, avg_glove=True)
X_test = utils_classic.encode_glove(glove_dict, X_test, avg_glove=True)
X_train = utils_classic.concatenate_sentences_arrays(X_train)
X_val = utils_classic.concatenate_sentences_arrays(X_val)
X_test = utils_classic.concatenate_sentences_arrays(X_test)

#doc2vec
utils_classic.train_doc2vec(X_train, 1, 0.02, 0.0002, 15, "doc2vec_model")
X_train = utils_classic.encode_doc2vec(X_train, "doc2vec_model")
X_val = utils_classic.encode_doc2vec(X_val, "doc2vec_model")
X_test = utils_classic.encode_doc2vec(X_test, "doc2vec_model")
X_train = utils_general.concatenate_sentences_arrays(X_train)
X_val = utils_general.concatenate_sentences_arrays(X_val)
X_test = utils_general.concatenate_sentences_arrays(X_test)

#lenghts of sentences
X_train=utils_classic.diff_of_length_sentences(X_train)
X_val=utils_classic.diff_of_length_sentences(X_val)
X_test=utils_classic.diff_of_length_sentences(X_test)

#bow
X_train,X_val,X_test=utils_classic.vectorizer_features(X_train[:5],X_val[:5],X_test[:5])

#3-ngram 
X_train,X_val,X_test=utils_classic.vectorizer_features(X_train[:5],X_val[:5],X_test[:5],type="char",ngram_range=(3,3),max_features=20000)
"""

print("LinearSVM")
svm = LinearSVC()
svm.fit(X_train, y_train)
y_pred_val = svm.predict(X_val)
y_pred_test = svm.predict(X_test)
print(accuracy_score(y_val, y_pred_val))
print(accuracy_score(y_test, y_pred_test))

print("LogReg")
logreg=LogisticRegression()
svm.fit(X_train, y_train)
y_pred_val = svm.predict(X_val)
y_pred_test = svm.predict(X_test)
print(accuracy_score(y_val, y_pred_val))
print(accuracy_score(y_test, y_pred_test))

