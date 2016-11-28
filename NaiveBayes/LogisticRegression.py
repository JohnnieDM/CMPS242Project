import numpy as np
import pandas as pd
from sklearn import linear_model
import pickle
from scipy.sparse import csr_matrix

data = pd.read_pickle('TF_IDFData.pickle')
testData = pd.read_pickle('TF_IDFData2.pickle')
wordsIndex = pickle.load(open('wordsIndex.pickle','rb'))

row = []
col = []
value = []
tf_idf = data['TF_IDF']
i = 0
for element in tf_idf:
    for k, v in element.items():
        row.append(i)
        col.append(wordsIndex[k])
        value.append(v)
    i += 1

X = csr_matrix((value, (row, col)), shape=(tf_idf.count(), len(wordsIndex)))
Y = data['sentiment'].as_matrix()


row = []
col = []
value = []
testTf_idf = testData['TF_IDF']
i = 0
for element in tf_idf:
    if k in wordsIndex:
        for k, v in element.items():
            row.append(i)
            col.append(wordsIndex[k])
            value.append(v)
    i += 1

testX = csr_matrix((value, (row, col)), shape=(tf_idf.count(), len(wordsIndex)))
testY = testData['sentiment'].as_matrix()

logreg = linear_model.LogisticRegression(dual=True, C=1e-9)
logreg.fit(X, Y)
predictY = logreg.predict(testX)
predictY[predictY == testY].shape
