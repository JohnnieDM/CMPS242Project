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
        if 'uni_tokens' in row:
            uni_tokens = row['uni_tokens']
            # algorithm
            for word in uni_tokens:
                if word in bow[1]:
                    sumOfClass1 += math.log(bow[1][word], 10)
                else:
                    sumOfClass1 += math.log(1.0 / (uni_n_words_[1] + uni_unique_words), 10)

                if word in uni_bow[0]:
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

                if word in uni_bow[0]:
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
        print "hi"
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

    # print accuracy
    print "NaiveBayes Result:"
    print "True Positive:  ", predictLabel[predictLabel == 1][predictLabel == testData['sentiment']].count()
    print "False Positive: ", predictLabel[predictLabel == 1][predictLabel != testData['sentiment']].count()
    print "True Negative:  ", predictLabel[predictLabel == 0][predictLabel == testData['sentiment']].count()
    print "False Negative: ", predictLabel[predictLabel == 0][predictLabel != testData['sentiment']].count()
    return predictLabel


def logistic_regression(data, testData, wordsIndex, args_tf_idf):
    row = []
    col = []
    value = []
    if args_tf_idf:
        frequency = data['TF_IDF']
    else:
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
    if args_tf_idf:
        frequency = testData['TF_IDF']
    else:
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

    logreg = linear_model.LogisticRegression(dual=True, C=1e-9)
    logreg.fit(X, Y)
    predictLabel = logreg.predict(testX)
    print "Logistic Regression Result:"
    true = predictLabel[predictLabel == testY]
    false = predictLabel[predictLabel != testY]
    print "True Positive:  ", true[true == 1].shape[0]
    print "False Positive: ", false[false == 1].shape[0]
    print "True Negative:  ", true[true == 0].shape[0]
    print "False Negative: ", false[false == 0].shape[0]
    return predictLabel

def init():
    parser = argparse.ArgumentParser(description="Specify feature types")
    parser.add_argument("-u", "--unigram", help="activate the unigram feature",
                      action="store_true")
    parser.add_argument("-b", "--bigram", help="activate the bigram feature",
                      action="store_true")
    parser.add_argument("-l", "--liwc", help="activate the LIWC feature",
                      action="store_true")
    parser.add_argument("-a", "--all", help="create maximum combos and pickle each",
                      action="store_true")
    parser.add_argument("-t", "--tfidf", help="use tfidf frequency count",
                      action="store_true")
    parser.add_argument("-nb", "--naive_bayes", help="use naive bayes classifier",
                      action="store_true")
    parser.add_argument("-lr", "--logistic_regression", help="use naive bayes classifier",
                      action="store_true")
    return parser.parse_args()
if( __name__ == '__main__'):
    #load feature vectors
    args = init()
    data = pd.read_pickle('jar_of_/train_features-a.pkl')
    testData = pd.read_pickle('jar_of_/test_features-a.pkl')
    wordsIndex = pickle.load(open('jar_of_/train_wordsIndex-a.pkl'))
    if args.naive_bayes:
        nbTestData = naive_bayes(data, testData)
    if args.logistic_regression:
        lrTestData = logistic_regression(data, testData, wordsIndex, args.tfidf)



