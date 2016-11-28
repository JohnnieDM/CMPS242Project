import os
import random
import operator
import math
import re
import pandas as pd
import numpy as np
import word_category_counter

def countWords(row):
	Class = row[1]
	for word in row[0]:
		bow[Class][word] = bow[Class].get(word,0) + 1.0
		N_words_[Class]+=1

def applyNaiveBayes(row):
	Class = row[1]
	B = 0
	C = 0
	#algorithm
	for word in row[0]:
		if word in bow[1]:
			B += math.log(bow[1][word],10)
		else:
			B += math.log(1.0/(N_words_[1]+wordB),10)

		if word in bow[0]:
			C += math.log(bow[0][word],10)
		else:
			C += math.log(1.0/(N_words_[0]+wordB),10)
	#print A,B,C
	#classification
	t=-1
	if A+B-C >0:
		t=1
	else:
		t=0
	#accuracy
	global TruePositive
	global TrueNegative
	global FalsePositive
	global FalseNegative
	if t==Class:
		if t==1:
			TruePositive+=1
		else:
			TrueNegative+=1
	else:
		if t==1:
			FalsePositive+=1
		else:
			FalseNegative+=1

if( __name__ == '__main__'):
	#load feature vectors
	Data = pd.read_pickle('NaiveBayesData.pkl')
	NBData = Data[['text','sentiment']]
	NBData_1 = NBData[NBData.sentiment==1]
	NBData_0 = NBData[NBData.sentiment==0]


	bow = [{},{}]
	#number of docs
	N = NBData.count()[0]
	#number of docs in each class
	N_ = [NBData_0.count()[0], NBData_1.count()[0]]

	#prior
	Prior = [N_[0]/float(N), N_[1]/float(N)]
	##total words for each class
	N_words_ = [0, 0]

	NBData.apply(countWords, axis = 1)

	#unique words for all classes
	wordB=len(set(bow[0].keys()+bow[1].keys()))
	#laplace smoothing
	for i in range(len(bow)):
		for word in bow[i]:
			bow[i][word]=(float(bow[i][word])+1)/(N_words_[i]+wordB)

	#generate test data
	TestData = pd.read_csv('NaiveBayesData2.csv')

	A = math.log(Prior[1],10)-math.log(Prior[0],10)
	TruePositive = 0
	FalsePositive = 0
	TrueNegative = 0
	FalseNegative = 0
	# A -> prior , B -> sum of the class 1, C -> sum of the class 0
	#run test
	NBTestData = TestData[['text','sentiment']]
	NBTestData.apply(applyNaiveBayes, axis=1)

	#print accuracy
	print "True Positive:  ", TruePositive
	print "False Positive: ", FalsePositive
	print "True Negative:  ", TrueNegative
	print "False Negative: ", FalseNegative
