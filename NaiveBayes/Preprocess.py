import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
import time
import sys  
import argparse

stops = set([str(w) for w in stopwords.words('english')])

def get_frequencies(dict):
  """
  Place holder for the different frequencies we will use.
  """
  r = {}
  for key, ngrams in dict.items():
    r[key] = [1 for ngram in ngrams]
  return r

def generate_ngram_feats(unigram_activated, bigram_activated, review):
  """
  Generate the n-gram features that are activated.
  """
  if not (unigram_activated and bigram_activated):
    return

  texts = review['text'].to_dict()
  unigram_dict = {}
  bigram_dict = {}
  count = 0
  size = len(texts.items())
  unique_tokens = set()
  for key, text in texts.items():
    count+=1
    if count % 1000 == 1:
      sys.stdout.write("%2.f" % (100.0 * count/size) + '%, completed: '+str(count)+'/'+str(size)+'\r')
      sys.stdout.flush()
    unigrams = []
    bigrams = []
    for sent in nltk.sent_tokenize(text):
      tokens = []
      for word in nltk.word_tokenize(sent):
        word = word.lower()
        if word not in stops:
          unigrams.append(word)
          tokens.append(word)
          unique_tokens.add(word)
      bigrams.append(list(ngrams(tokens,2))) # list() to unpack the bigram generator object
    unigram_dict[key] = unigrams
    bigram_dict[key] = bigrams

  if unigram_activated:
    review['unigrams'] = pd.Series(get_frequencies(unigram_dict))
  if bigram_activated:
    review['bigrams'] = pd.Series(get_frequencies(bigram_dict))

  return all_tokens


def generate_liwc_feature(activated):
  """
  Generate LIWC features
  """
  pass




if __name__ == "__main__":
  # Each argument is parsed as a boolean which defaults to False when not given.
  parser = argparse.ArgumentParser(description="Specify feature types")
  parser.add_argument("-u", "--unigram", help="activate the unigram feature",
                    action="store_true")
  parser.add_argument("-b", "--bigram", help="activate the bigram feature",
                    action="store_true")
  parser.add_argument("-l", "--liwc", help="activate the LIWC feature",
                    action="store_true")
  args = parser.parse_args()
  if args.unigram:
    print "unigram feature activated"
  if args.bigram:
    print "bigram feature activated"
  if args.liwc:
    print "LIWC feature activated"

  # Read in the CSV of reviews, only using 1% of the data.
  review = pd.read_csv('yelp_academic_dataset_review.csv', encoding='utf-8')[['business_id','stars','text']].sample(frac=0.01, replace=False)
  print "read_csv completed"

  # Feature: Generate unigram and bigram features if activated.
  # Also collect set of tokens in all reviews.
  unique_tokens = generate_ngram_feats(args.unigram, args.bigram, review)

  sorted(list(unique_tokens))

  # Feature: Generate LIWC features if activated.
  generate_liwc_feature(args.liwc)

  # Read the business CSV here to avoid memory problem.
  business = pd.read_csv('yelp_academic_dataset_business.csv', low_memory=False, encoding='utf-8')[['business_id','name','categories']]

  # Process categories: use the most frequent to be the category for each review.
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

  # Label: positive 1, negative 0.
  # Define positive to be >2 stars, negative to be <=2 stars.
  sentiment = []
  for s in star:
    if s<=2:
      sentiment.append(0)
    else:
      sentiment.append(1)
  business['categories'] = newCategories

  # Merge business and review DataFrames.
  mergeBusRev = pd.merge(business, review, on = 'business_id')
  NaiveBayesData = pd.concat([mergeBusRev, pd.DataFrame({'sentiment':sentiment})],axis = 1)

  # Save the DataFrame with features to pickle.
  NaiveBayesData.to_pickle('features' +''.join(sorted(sys.argv[1:]))+ '.pickle')

