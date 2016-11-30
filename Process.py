import os
import random
import operator
import math
import re
import pandas as pd
import numpy as np
import word_category_counter
import argparse
from sklearn import linear_model
import pickle
from scipy.sparse import csr_matrix
import sys
import word_category_counter
from sklearn import svm

def naive_bayes(data, testData):

    def count_uni_words(row):
        Class = row[1]
        for word in row[0]:
            uni_bow[Class][word] = uni_bow[Class].get(word, 0) + 1.0
            uni_n_words_[Class] += 1
    def count_bi_words(row):
        Class = row[1]
        for word in row[0]:
            bi_bow[Class][word] = bi_bow[Class].get(word, 0) + 1.0
            bi_n_words_[Class] += 1

    def apply_naive_bayes(row):
        sumOfClass1 = 0
        sumOfClass2 = 0
        if 'liwc' in row:
            liwc_scores = word_category_counter.score_text(row['text'])
            # liwc_dict[key] = liwc_scores
            negative_score = liwc_scores["Negative Emotion"]
            positive_score = liwc_scores["Positive Emotion"]
            if positive_score > negative_score:
                sumOfClass1 += bow[1]['liwc:positive']
            else:
                sumOfClass2 += bow[0]["liwc:negative"]

        if 'uni_tokens' in row:
            uni_tokens = row['uni_tokens']
            # algorithm
            for word in uni_tokens:
                if word in bow[1]:
                    sumOfClass1 += math.log(bow[1][word], 10)
                else:
                    sumOfClass1 += math.log(1.0 / (uni_n_words_[1] + uni_unique_words), 10)

                if word in bow[0]:
                    sumOfClass2 += math.log(bow[0][word], 10)
                else:
                    sumOfClass2 += math.log(1.0 / (uni_n_words_[0] + uni_unique_words), 10)
        if 'bi_tokens' in row:
            bi_tokens = row['bi_tokens']
            # algorithm
            for word in bi_tokens:
                if word in bow[1]:
                    sumOfClass1 += math.log(bow[1][word], 10)
                else:
                    sumOfClass1 += math.log(1.0 / (bi_n_words_[1] + bi_unique_words), 10)

                if word in bow[0]:
                    sumOfClass2 += math.log(bow[0][word], 10)
                else:
                    sumOfClass2 += math.log(1.0 / (bi_n_words_[0] + bi_unique_words), 10)
        # print A,B,C
        # classification
        if priorPortion + sumOfClass1 - sumOfClass2 > 0:
            t = 1
        else:
            t = 0
        return t

    # number of docs
    n = data.count()[0]
    data_1 = data[data.sentiment == 1]
    data_0 = data[data.sentiment == 0]
    # number of docs in each class
    n_ = [data_0.count()[0], data_1.count()[0]]
    # prior
    prior = [n_[0] / float(n), n_[1] / float(n)]

    bow = [{},{}]
    if 'liwc' in data.columns:
        bow[0].update({"liwc:negative":1})
        bow[1].update({"liwc:positive": 1})
    if 'uni_tokens' in data.columns:
        nb_data = data[['uni_tokens', 'sentiment']]


        uni_bow = [{}, {}]

        ##total words for each class
        uni_n_words_ = [0, 0]

        nb_data.apply(count_uni_words, axis=1)

        # unique words for all classes
        uni_unique_words = len(set(uni_bow[0].keys() + uni_bow[1].keys()))
        # laplace smoothing
        for i in range(len(uni_bow)):
            for word in uni_bow[i]:
                uni_bow[i][word] = (float(uni_bow[i][word]) + 1) / (uni_n_words_[i] + uni_unique_words)
        bow[0].update(uni_bow[0])
        bow[1].update(uni_bow[1])

    if 'bi_tokens' in data.columns:
        nb_data = data[['bi_tokens', 'sentiment']]

        bi_bow = [{}, {}]
        # number of docs

        ##total words for each class
        bi_n_words_ = [0, 0]

        nb_data.apply(count_bi_words, axis=1)

        # unique words for all classes
        bi_unique_words = len(set(bi_bow[0].keys() + bi_bow[1].keys()))
        # laplace smoothing
        for i in range(len(bi_bow)):
            for word in bi_bow[i]:
                bi_bow[i][word] = (float(bi_bow[i][word]) + 1) / (bi_n_words_[i] + bi_unique_words)
        bow[0].update(bi_bow[0])
        bow[1].update(bi_bow[1])

    # generate test data

    priorPortion = math.log(prior[1], 10) - math.log(prior[0], 10)
    # A -> prior , B -> sum of the class 1, C -> sum of the class 0
    # run test

    predictLabel = testData.apply(apply_naive_bayes, axis=1)
    disc_feats_NB(bow)
    print_result(predictLabel.as_matrix(), testData['sentiment'].as_matrix())
    return predictLabel


def print_result(predictLabel, testLabel):
    """
    Calculate and print the accuracies (accuracy, precision, recall, and F1 score).
    """
    true = predictLabel[predictLabel == testLabel]
    false = predictLabel[predictLabel != testLabel]
    tp = true[true == 1].shape[0] + 0.0
    fp = false[false == 1].shape[0] + 0.0
    tn = true[true == 0].shape[0] + 0.0
    fn = false[false == 0].shape[0] + 0.0
    accuracy = (tp+tn) / (tp+fp+tn+fn)
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1 = 2*((precision*recall) / (precision+recall))
    print "====================================================="
    print "Results:"
    print "  Accuracy:\t", accuracy
    print "  Precision:\t", precision
    print "  Recall:\t", recall
    print "  F1 score:\t", f1
    print "====================================================="
def generate_matrix(data, testData, wordsIndex):
    row = []
    col = []
    value = []

    frequency = data['frequency']
    i = 0
    for element in frequency:
        for k, v in element.items():
            row.append(i)
            col.append(wordsIndex[k])
            value.append(v)
        i += 1

    X = csr_matrix((value, (row, col)), shape=(frequency.count(), len(wordsIndex)))
    Y = data['sentiment'].as_matrix()

    row = []
    col = []
    value = []
    frequency = testData['frequency']
    i = 0
    for element in frequency:
        for k, v in element.items():
            if k in wordsIndex:
                row.append(i)
                col.append(wordsIndex[k])
                value.append(v)
        i += 1

    testX = csr_matrix((value, (row, col)), shape=(frequency.count(), len(wordsIndex)))
    testY = testData['sentiment'].as_matrix()
    return X, Y, testX, testY


def generate_svm(data, testData, wordsIndex):
    X, Y, testX, testY = generate_matrix(data, testData, wordsIndex)

    penalty = 1.

    gamma_svm = 1.

    # Training and testing for each value of gamma
    #clf = svm.SVC(kernel='rbf', C=penalty, gamma=gamma_svm)
    print(' svm is fitting now...')
    clf = svm.SVC(kernel='linear')
    clf.fit(X, Y)
    predictLabel = clf.predict(testX)
    print_result(predictLabel, testY)
    return predictLabel


def logistic_regression(data, testData, wordsIndex, revWordsIndex):
    X, Y, testX, testY = generate_matrix(data, testData, wordsIndex)

    logreg = linear_model.LogisticRegression(dual=True, C=1e-9)
    logreg.fit(X, Y)

    coef = logreg.coef_[0]
    coef = coef.tolist()
    coef = [(i,coef[i]) for i in range(len(coef))]
    coef = sorted(coef, key=lambda x: x[1], reverse=True)
    print "The 10 most discriminative words"
    print revWordsIndex[coef[0][0]], revWordsIndex[coef[1][0]], revWordsIndex[coef[2][0]], revWordsIndex[coef[3][0]],\
        revWordsIndex[coef[4][0]], revWordsIndex[coef[5][0]], revWordsIndex[coef[6][0]], revWordsIndex[coef[7][0]],\
        revWordsIndex[coef[8][0]], revWordsIndex[coef[9][0]]
    predictLabel = logreg.predict(testX)
    print_result(predictLabel, testY)
    return predictLabel

def disc_feats_NB(training):
    ham = [(k, training[0][k]) for k in sorted(training[0], key=training[0].get, reverse=True)]
    spam = [(k, training[1][k]) for k in sorted(training[1], key=training[1].get, reverse=True)]

    ham = dict(ham[:100])
    spam = dict(spam[:100])

    rm_key = []

    for k in ham.keys():
        if k in spam.keys():
            rm_key.append(k)

    for rm in rm_key:
        del ham[rm]
        del spam[rm]

    ham = [(k, ham[k]) for k in sorted(ham, key=ham.get, reverse=True)]
    spam = [(k, spam[k]) for k in sorted(spam, key=spam.get, reverse=True)]

    print("Top positive features:")
    print(spam)
    print("Top negative features:")
    print(ham)


def init():
    parser = argparse.ArgumentParser(description="Specify feature types")
    # Options to select the classifier:
    parser.add_argument("-c", "--classifier", help="the type of classifier to train (nb for Naive Bayes,\
                        lr for Logistic Regression, and svm for Support Vector Machine)", required=True)
    parser.add_argument("-d", "--document", help="the exact document for running classifier", required=True)
    return parser.parse_args()

if( __name__ == '__main__'):
    #load feature vectors
    args = init()
    path = args.document
    train_data = pd.read_pickle(os.path.join(path, 'train_features.pkl'))
    test_data = pd.read_pickle(os.path.join(path, 'test_features.pkl'))
    words_index = pickle.load(open(os.path.join(path, 'wordsIndex.pkl')))
    rev_words_index = pickle.load(open(os.path.join(path, 'revWordsIndex.pkl')))

    if args.classifier == "nb":
        nbTestData = naive_bayes(train_data, test_data)
    if args.classifier == "lr":
        lrTestData = logistic_regression(train_data, test_data, words_index, rev_words_index)
    if args.classifier == "svm":
        svmTestData = generate_svm(train_data, test_data, words_index)


