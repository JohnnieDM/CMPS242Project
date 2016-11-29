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

def naive_bayes(data, testData, args):
    def countWords(row):
        Class = row[1]
        for word in row[0]:
            bow[Class][word] = bow[Class].get(word, 0) + 1.0
            n_words_[Class] += 1

    def applyNaiveBayes(row):
        sumOfClass1 = 0
        sumOfClass2 = 0
        # algorithm
        for word in row[0]:
            if word in bow[1]:
                sumOfClass1 += math.log(bow[1][word], 10)
            else:
                sumOfClass1 += math.log(1.0 / (n_words_[1] + uniqueWords), 10)

            if word in bow[0]:
                sumOfClass2 += math.log(bow[0][word], 10)
            else:
                sumOfClass2 += math.log(1.0 / (n_words_[0] + uniqueWords), 10)
        # print A,B,C
        # classification
        if priorPortion + sumOfClass1 - sumOfClass2 > 0:
            t = 1
        else:
            t = 0
        return t

    nbData = data[['uni_tokens', 'sentiment']]
    nbData_1 = nbData[nbData.sentiment == 1]
    nbData_0 = nbData[nbData.sentiment == 0]

    bow = [{}, {}]
    # number of docs
    n = nbData.count()[0]
    # number of docs in each class
    n_ = [nbData_0.count()[0], nbData_1.count()[0]]

    # prior
    prior = [n_[0] / float(n), n_[1] / float(n)]
    ##total words for each class
    n_words_ = [0, 0]

    nbData.apply(countWords, axis=1)

    # unique words for all classes
    uniqueWords = len(set(bow[0].keys() + bow[1].keys()))
    # laplace smoothing
    for i in range(len(bow)):
        for word in bow[i]:
            bow[i][word] = (float(bow[i][word]) + 1) / (n_words_[i] + uniqueWords)

    # generate test data

    priorPortion = math.log(prior[1], 10) - math.log(prior[0], 10)
    # A -> prior , B -> sum of the class 1, C -> sum of the class 0
    # run test
    nbTestData = testData[['uni_tokens', 'sentiment']]
    predictLabel = nbTestData.apply(applyNaiveBayes, axis=1)
    disc_feats_NB(bow)

    # Print accuracies
    print_result(predictLabel)
    #print "Unigrm NaiveBayes Result:"
    #print "True Positive:  ", predictLabel[predictLabel == 1][predictLabel == nbTestData['sentiment']].count()
    #print "False Positive: ", predictLabel[predictLabel == 1][predictLabel != nbTestData['sentiment']].count()
    #print "True Negative:  ", predictLabel[predictLabel == 0][predictLabel == nbTestData['sentiment']].count()
    #print "False Negative: ", predictLabel[predictLabel == 0][predictLabel != nbTestData['sentiment']].count()
    return predictLabel

def print_result(predictLabel):
    """
    Calculate and print the accuracies (accuracy, precision, recall, and F1 score).
    """
    tp = predictLabel[predictLabel == 1][predictLabel == nbTestData['sentiment']].count()
    fp = predictLabel[predictLabel == 1][predictLabel != nbTestData['sentiment']].count()
    tn = predictLabel[predictLabel == 0][predictLabel == nbTestData['sentiment']].count()
    fn = predictLabel[predictLabel == 0][predictLabel != nbTestData['sentiment']].count()
    accuracy = (tp+tn) / (tp+fp+tn+fn)
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1 = 2((precision*recall) / (precision+recall))
    print("=====================================================")
    print("Results:")
    print("  Accuracy: ", accuracy)
    print("  Precision: ", precision)
    print("  Recall: ", recall)
    print("  F1 score: ", f1)

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
    # Options to select the classifier:
    parser.add_argument("-nb", "--naive_bayes", help="use naive bayes classifier",
                      action="store_true")
    parser.add_argument("-lr", "--logistic_regression", help="use naive bayes classifier",
                      action="store_true")
    parser.add_argument("-svm", "--SVM", help="use support vector classifier",
                      action="store_true")
    parser.add_argument("-t", "--train", help="the name of the pickled feature file")
    parser.add_argument("-s", "--test", help="the name of the pickled feature file")
    parser.add_argument("-w", "--wordsindex", help="the name of the pickled feature file")
    return parser.parse_args()

if( __name__ == '__main__'):
    #load feature vectors
    args = init()
    train_data = pd.read_pickle(args.train)
    test_data = pd.read_pickle(args.test)
    wordsIndex = pickle.load(open(args.words_index))

    if args.naive_bayes:
        nbTestData = naive_bayes(train_data, test_Data, args)



