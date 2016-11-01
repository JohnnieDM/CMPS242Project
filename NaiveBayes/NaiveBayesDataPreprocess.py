import pandas as pd
business = pd.read_csv('yelp_academic_dataset_business.csv',low_memory = False)[['business_id','name','categories']]
review = pd.read_csv('yelp_academic_dataset_review.csv')[['business_id','stars','text']]
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
NaiveBayesData.to_csv('NaiveBayesData.csv',index=False)

	
