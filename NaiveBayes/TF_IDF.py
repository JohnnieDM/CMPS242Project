import pandas as pd 
import numpy as np
import pickle
def calculateTF(row):
	TFDict = {}
	for word in row:
		TFDict[word] = TFDict.get( word, 0 ) + 1
	TFDict = { k: float(v)/len(row) for k,v in TFDict.iteritems()}
	return TFDict
IDFDict = {}
def calculateIDF(row):
	row = set(row)
	for word in row:
		IDFDict[word] = IDFDict.get( word, 0 ) + 1

def calculateTF_IDF(row):
	return {k : v * IDFDict[k] for k,v in row.items()}

DataName = 'NaiveBayesData.pickle'
data = pd.read_pickle(DataName)

N = data.shape[0]
TFDict = data['text'].apply(calculateTF)
data['text'].apply(calculateIDF)
IDFDict = {k: float(N)/v for k,v in IDFDict.items()}
TF_IDFDict = TFDict.apply(calculateTF_IDF).rename('TF_IDF')
TF_IDFData = pd.concat([data, TF_IDFDict],axis = 1)
TF_IDFData.to_pickle('TF_IDFData.pickle')
wordsIndex = {k: i for i, k in enumerate(IDFDict.keys())}
revWordsIndex = {i: k for i, k in enumerate(IDFDict.keys())}
pickle.dump(wordsIndex, open('wordsIndex.pickle', 'wb'))
pickle.dump(revWordsIndex, open('revWordsIndex.pickle', 'wb'))

DataName = 'NaiveBayesData2.pickle'
data = pd.read_pickle(DataName)

N = data.shape[0]
TFDict = data['text'].apply(calculateTF)
data['text'].apply(calculateIDF)
IDFDict = {k: float(N)/v for k,v in IDFDict.items()}
TF_IDFDict = TFDict.apply(calculateTF_IDF).rename('TF_IDF')
TF_IDFData = pd.concat([data, TF_IDFDict],axis = 1)
TF_IDFData.to_pickle('TF_IDFData2.pickle')
wordsIndex = {k: i for i, k in enumerate(IDFDict.keys())}
revWordsIndex = {i: k for i, k in enumerate(IDFDict.keys())}
pickle.dump(wordsIndex, open('wordsIndex2.pickle', 'wb'))
pickle.dump(revWordsIndex, open('revWordsIndex2.pickle', 'wb'))
