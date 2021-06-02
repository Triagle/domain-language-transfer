import csv
import pickle

import cleantext
import numpy as np
from gensim.models.word2vec import Word2Vec
from numpy.lib.function_base import average
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                  SGDClassifier)
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC

EN_CORPUS = 'eng-eu_web_2014_100K/eng-eu_web_2014_100K-sentences.txt'
DE_CORPUS = 'deu-eu_web_2015_100K/deu-eu_web_2015_100K-sentences.txt'
EN_DICTIONARY = 'google-10000-english.txt'
DE_DICTIONARY = 'google-10000-german.txt'


def clean_corpus(txt_file):
    with open(txt_file, 'r') as f:
        cln = cleantext.clean(f.read(), no_urls=True, no_numbers=True,
                              no_punct=True, no_currency_symbols=True)
    with open(txt_file, 'w') as f:
        f.write(cln)


def prepare_tweet(tweet: str):
    cleaned = cleantext.clean(tweet, no_numbers=True,
                              no_punct=True, no_currency_symbols=True)
    cleaned = cleaned.replace('url', '<url>')
    return cleaned


def clean_olid_data(txt_file: str):
    with open(txt_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = [{'id': r['id'], 'tweet': prepare_tweet(
            r['tweet']), 'subtask_a': r['subtask_a']} for r in reader]
    with open('olid-clean.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writerows(rows)


def learn_transformation_matrix(source_model, source_dictionary, target_model, target_dictionary):
    with open(source_dictionary, 'r') as f:
        source_words = [l.rstrip().lower() for l in f.readlines()]

    with open(target_dictionary, 'r') as f:
        target_words = [l.rstrip().lower() for l in f.readlines()]

    word_pairs = [(source_model.wv[source], target_model.wv[target]) for source, target in zip(
        source_words, target_words) if source in source_model.wv and target in target_model.wv]
    X = np.array([kv[0] for kv in word_pairs])
    Y = np.array([kv[1] for kv in word_pairs])
    reg = LinearRegression().fit(X, Y)
    return reg
    # clean_corpus(DE_CORPUS)


def translate(source_model: Word2Vec, mapping: LinearRegression, target_model: Word2Vec, word: str, topn=1):
    u = source_model.wv[word]
    v = mapping.predict(u.reshape((1, -1))).reshape((-1,))
    return target_model.wv.similar_by_vector(v, topn=topn)


def train_de_model():
    de_model = Word2Vec(corpus_file=DE_CORPUS,
                        vector_size=800, window=15, sg=1)
    de_model.save('sg-model-de-800.model')


def train_de_en_tfm():
    en_model = Word2Vec.load('sg-model-en.model')
    de_model = Word2Vec.load('sg-model-de-800.model')
    tfm = learn_transformation_matrix(
        de_model, DE_DICTIONARY, en_model, EN_DICTIONARY)
    with open('de-en-tfm', 'wb') as f:
        pickle.dump(tfm, f)


def embed(model, sentence):
    arr = np.array([model.wv[w]
                   for w in sentence.split() if w in model.wv])
    return np.sum(arr, axis=0).reshape((1, -1))


def read_dataset():

    en_model = Word2Vec.load('sg-model-en-800.model')
    with open('olid-clean.csv', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        embeds = []
        skipped_samples = set()
        for i, row in enumerate(rows):
            e = embed(en_model, row['tweet'])
            if e.shape == (1, 800):
                embeds.append(e)
            else:
                skipped_samples.add(i)
        texts = np.concatenate(embeds)
        off_labels = np.array([int(row['off'] == 'OFF')
                              for i, row in enumerate(rows) if i not in skipped_samples])
    return texts, off_labels


def train_classifier():
    texts, off_labels = read_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, off_labels, random_state=314)

    clf = make_pipeline(
        MinMaxScaler(), LogisticRegression(class_weight='balanced'))
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    f1 = f1_score(y_train, y_pred_train)
    matrix = confusion_matrix(y_train, y_pred_train)
    print(f'(train) F1 score: {f1}')
    print(matrix)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    print(f'(test) F1 score: {f1}')
    print(matrix)


if __name__ == '__main__':
    train_classifier()
