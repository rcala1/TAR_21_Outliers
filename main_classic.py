import utils_general
import utils_classic
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np

X_dataset, y_dataset = utils_general.get_Xy_train(
    utils_general.load_dataset_numpy("train.csv")
)
X_dataset, y_dataset = utils_general.rmv_float_and_nan(X_dataset, y_dataset)
X_dataset = utils_general.remove_punctuation(X_dataset)
X_train, y_train, X_val, y_val, X_test, y_test = utils_general.split_train_dev_test(
    X_dataset, y_dataset
)

# glove
glove_dict = utils_classic.load_glove_dict("sst_glove_6b_300d.txt")
X_train_glove = utils_classic.encode_glove(glove_dict, X_train, avg_glove=True)
X_val_glove = utils_classic.encode_glove(glove_dict, X_val, avg_glove=True)
X_test_glove = utils_classic.encode_glove(glove_dict, X_test, avg_glove=True)
X_train_glove = utils_general.concatenate_sentences_arrays(X_train_glove)
X_val_glove = utils_general.concatenate_sentences_arrays(X_val_glove)
X_test_glove = utils_general.concatenate_sentences_arrays(X_test_glove)

"""
#doc2vec not useful
utils_classic.train_doc2vec(X_train, 1, 0.02, 0.0002, 15, "doc2vec_model")
X_train_doc = utils_classic.encode_doc2vec(X_train, "doc2vec_model")
X_val_doc = utils_classic.encode_doc2vec(X_val, "doc2vec_model")
X_test_doc = utils_classic.encode_doc2vec(X_test, "doc2vec_model")
X_train_doc = utils_general.concatenate_sentences_arrays(X_train_doc)
X_val_doc = utils_general.concatenate_sentences_arrays(X_val_doc)
X_test_doc = utils_general.concatenate_sentences_arrays(X_test_doc)
"""

# bow
X_train_bow, X_val_bow, X_test_bow = utils_classic.vectorizer_features(
    X_train, X_val, X_test, max_features=5000
)

# 2,3-char ngram
X_train_char, X_val_char, X_test_char = utils_classic.vectorizer_features(
    X_train, X_val, X_test, type="char", ngram_range=(2, 3), max_features=5000
)

X_train_char = np.array(X_train_char)
X_val_char = np.array(X_val_char)
X_test_char = np.array(X_test_char)

X_train_bow = np.array(X_train_bow)
X_val_bow = np.array(X_val_bow)
X_test_bow = np.array(X_test_bow)

X_train_glove = np.array(X_train_glove)
X_val_glove = np.array(X_val_glove)
X_test_glove = np.array(X_val_glove)

X_train = np.concatenate((X_train_char, X_train_bow, X_train_glove), axis=1)
X_val = np.concatenate((X_val_char, X_val_bow, X_val_glove), axis=1)
X_test = np.concatenate((X_test_char, X_test_bow, X_test_glove), axis=1)

print("LinearSVM")
svm = LinearSVC()
svm.fit(X_train, y_train)
y_pred_val = svm.predict(X_val)
y_pred_test = svm.predict(X_test)
print(accuracy_score(y_val, y_pred_val))
print(accuracy_score(y_test, y_pred_test))
