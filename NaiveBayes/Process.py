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

def naive_bayes(data, testData):
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

	# print accuracy
	print "Unigrm NaiveBayes Result:"
	print "True Positive:  ", predictLabel[predictLabel == 1][predictLabel == nbTestData['sentiment']].count()
	print "False Positive: ", predictLabel[predictLabel == 1][predictLabel != nbTestData['sentiment']].count()
	print "True Negative:  ", predictLabel[predictLabel == 0][predictLabel == nbTestData['sentiment']].count()
	print "False Negative: ", predictLabel[predictLabel == 0][predictLabel != nbTestData['sentiment']].count()
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
	parser.add_argument("-un", "--uni_naive_bayes", help="use naive bayes classifier",
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
	if args.uni_naive_bayes:
		nbTestData = naive_bayes(data, testData, wordsIndex)
	if args.logistic_regression:
		lrTestData = logistic_regression(data, testData, wordsIndex, args.tfidf)


