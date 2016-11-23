import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import time
import sys  

'''
Fix for
UnicodeDecodeError: 'ascii' codec can't decode byte 0xc3 in position 4: ordinal not in range(128)
'''

review = pd.read_csv('yelp_academic_dataset_review.csv', encoding='utf-8')[['business_id','stars','text']].sample(frac=0.01, replace=False)

print "read_csv completed."

texts = review['text'].to_dict()
texts2 = {}
count=0
size = len(texts.items())
stops = set([str(w) for w in stopwords.words('english')])

for key, text in texts.items():
  count+=1
  if count % 1000 == 1:
    sys.stdout.write("%2.f" % (100.0 * count/size) + '%, completed: '+str(count)+'/'+str(size)+'\r')
    sys.stdout.flush()
  texts2[key] = [word.lower() for word in re.findall('[^\d\W]+', text) if word.lower() not in stops]
review['text'] = pd.Series(texts2)

#bids = review['business_id'].get_values()
business = pd.read_csv('yelp_academic_dataset_business.csv', low_memory=False, encoding='utf-8')[['business_id','name','categories']]
#business = business[(business['business_id'].isin(bids))]


categories = business['categories']
star = review['stars']
countCate = {}
for c in categories:
  cateList = c.split(",")
  for cate in cateList:
    element = cate.strip(" '[]")
    countCate[element] = countCate.get(element,0)+1
newCategories = []
for c in categories:
  cateList = c.split(",")
  maxCate = 0
  finalCate = ""
  for cate in cateList:
    element = cate.strip(" '[]")
    if maxCate<countCate[element]:
      finalCate = element
      maxCate = countCate[element]
  newCategories.append(finalCate)
sentiment = []
for s in star:
  if s<=2:
    sentiment.append(0)
  else:
    sentiment.append(1)
business['categories'] = newCategories
mergeBusRev = pd.merge(business, review, on = 'business_id')
NaiveBayesData = pd.concat([mergeBusRev, pd.DataFrame({'sentiment':sentiment})],axis = 1)
NaiveBayesData.to_pickle('NaiveBayesData.pickle')

