import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import utils_general


def get_largest_length(dataset):
    length = 0
    for i in range(len(dataset)):
        if len(dataset[i]) > length:
            length = len(dataset[i])
    return length


def pad_dataset(dataset, length):
    dataset_new = []
    for i in range(len(dataset)):
        dataset_new += [np.array(dataset[i].copy())]
        dataset_new[i].resize(length)
    return dataset_new


def load_glove_dict(file_path):
    f = open(file_path, "r")
    glove_dict = {}
    for line in f.read().splitlines():
        word, embedding = line.split(" ", 1)
        glove_dict[word] = np.array([float(val) for val in embedding.split()])
    glove_dict["<UNK>"] = np.random.randn(300)
    return glove_dict


def encode_glove(glove_dict, dataset, avg_glove=False):

    dataset = utils_general.split_sentences_in_tokens(dataset)

    glove_encoded = []
    for i in range(len(dataset)):
        glove_encoded += [[[], []]]
        for j in range(len(dataset[i][0])):
            vector = glove_dict.get(dataset[i][0][j])
            if vector is not None:
                glove_encoded[i][0] += [vector]
            else:
                glove_encoded[i][0] += [glove_dict["<UNK>"]]
        for j in range(len(dataset[i][1])):
            vector = glove_dict.get(dataset[i][1][j])
            if vector is not None:
                glove_encoded[i][1] += [vector]
            else:
                glove_encoded[i][1] += [glove_dict["<UNK>"]]

    if avg_glove:
        for i in range(len(glove_encoded)):
            if not glove_encoded[i][0]:
                glove_encoded[i][0] = np.ones(300)
            else:
                glove_encoded[i][0] = np.mean(glove_encoded[i][0], axis=0)
            if not glove_encoded[i][1]:
                glove_encoded[i][1] = np.ones(300)
            else:
                glove_encoded[i][1] = np.mean(glove_encoded[i][1], axis=0)
    return glove_encoded


def train_doc2vec(dataset, vec_size, alpha, min_alpha, epochs, save_file_path):

    tagged_documents_first = [
        TaggedDocument(word_tokenize(d.lower()), [str(i)])
        for i, d in enumerate(dataset[:][0])
    ]

    tagged_documents_second = [
        TaggedDocument(word_tokenize(d.lower()), [str(i)])
        for i, d in enumerate(dataset[:][1])
    ]

    tagged_documents = tagged_documents_first + tagged_documents_second

    model = Doc2Vec(
        documents=tagged_documents,
        vector_size=vec_size,
        alpha=alpha,
        min_alpha=min_alpha,
        min_count=1,
        epochs=epochs,
    )
    model.save(save_file_path)


def encode_doc2vec(dataset, load_file_path):

    model = Doc2Vec.load(load_file_path)

    doc2vec_encoded = []
    for i in range(len(dataset)):
        doc2vec_encoded += [[]]
        doc2vec_encoded[i] += [model.infer_vector(word_tokenize(dataset[i][0].lower()))]
        doc2vec_encoded[i] += [model.infer_vector(word_tokenize(dataset[i][1].lower()))]

    return doc2vec_encoded


def length_of_sentences(dataset):

    lengths = []
    for i in range(len(dataset)):
        lengths += [[]]
        lengths[i] += [len(dataset[i][0])]
        lengths[i] += [len(dataset[i][1])]
    return lengths


def diff_of_length_sentences(dataset):

    difference_length = []
    for i in range(len(dataset)):
        difference_length += [[]]
        difference_length[i] += [len(dataset[i][0]) - len(dataset[i][1])]
        difference_length[i] += [
            len(word_tokenize(dataset[i][0])) - len(word_tokenize(dataset[i][1]))
        ]

    return difference_length


def vectorizer_features(
    train_dataset,
    val_dataset,
    test_dataset,
    type="word",
    ngram_range=(1, 1),
    max_features=None,
):

    train_dataset_concatenated = utils_general.concatenate_sentences(train_dataset)
    countvectorizer = CountVectorizer(
        analyzer=type, ngram_range=ngram_range, max_features=max_features
    )
    countvectorizer.fit(train_dataset_concatenated)

    first_sentences_train, second_sentences_train = zip(*train_dataset)
    first_sentences_val, second_sentences_val = zip(*val_dataset)
    first_sentences_test, second_sentences_test = zip(*test_dataset)

    first_sentences_train = countvectorizer.transform(first_sentences_train).toarray()
    second_sentences_train = countvectorizer.transform(second_sentences_train).toarray()
    concatenated_train = [
        list(first) + list(second)
        for first, second in zip(first_sentences_train, second_sentences_train)
    ]

    first_sentences_val = countvectorizer.transform(first_sentences_val).toarray()
    second_sentences_val = countvectorizer.transform(second_sentences_val).toarray()
    concatenated_val = [
        list(first) + list(second)
        for first, second in zip(first_sentences_val, second_sentences_val)
    ]

    first_sentences_test = countvectorizer.transform(first_sentences_test).toarray()
    second_sentences_test = countvectorizer.transform(second_sentences_test).toarray()
    concatenated_test = [
        list(first) + list(second)
        for first, second in zip(first_sentences_test, second_sentences_test)
    ]

    return concatenated_train, concatenated_val, concatenated_test
