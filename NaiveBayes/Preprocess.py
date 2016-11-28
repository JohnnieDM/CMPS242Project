import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
import time
import sys  
import argparse
import word_category_counter
import os
import numpy as np

stops = set([str(w) for w in stopwords.words('english')])

def get_frequencies(df, tokens):
  """
  Place holder for the different frequencies we will use.
  """
  return pd.Series()

def generate_ngram_feats(unigram_activated, bigram_activated, review):
  """
  Generate the n-gram features that are activated.
  Add columns for unigram tokens and bigram tokens,
  compute the frequencies (unigram) and conditional frequencies (bigram),
  and add these as columns.
  Returns lists of unique tokens for unigrams and bigrams.
  """
  if not (unigram_activated or bigram_activated):
    return

  texts = review['text'].to_dict()
  unigram_dict = {}
  bigram_dict = {}
  count = 0
  size = len(texts.items())
  unique_uni = set()
  unique_bi  = set()
  for key, text in texts.items():
    count+=1
    if count % 1000 == 1:
      sys.stdout.write("%2.f" % (100.0 * count/size) + '%, completed: '+str(count)+'/'+str(size)+'\r')
      sys.stdout.flush()
    unigrams = []
    bigrams = []
    for sent in nltk.sent_tokenize(text):
      tokens = []
      for word in re.findall('[^_^\d\W]+', sent):
        word = word.lower()
        if word not in stops:
          unigrams.append(word)
          tokens.append(word)
          unique_uni.add(word)
      if bigram_activated:
        bigrams = list(ngrams(tokens,2)) # list() to unpack the bigram generator object

    unigram_dict[key] = unigrams
    bigram_dict[key] = bigrams
    [unique_uni.add(unigram) for unigram in unigrams]
    [unique_bi.add(bigram) for bigram in bigrams]

  # Add columns of n-gram tokens of each text.
  if unigram_activated:
    review['uni_tokens'] = pd.Series(unigram_dict)
  if bigram_activated:
    review['bi_tokens'] = pd.Series(bigram_dict)

  # Add columns of frequencies.
  # Turn sets of unique tokens into sorted lists for frequency computation.
  if unigram_activated:
    unique_uni = sorted(list(unique_uni))
    review['uni_freq'] = get_frequencies(review, unique_uni)
  if bigram_activated:
    unique_bi = sorted(list(unique_bi))
    review['bi_freq'] = get_frequencies(review, unique_bi)
  print review

  return (unique_uni, unique_bi)


def add_liwc_features(review):
  """
  Args:
      (string)text: some text input
      (dict)feature_vector: a dict of features

  Returns:
      Modified feature vector

  """
  # All possible keys to the scores start on line 269
  # of the word_category_counter.py script
  # for key in liwc_scores.keys():
  #     feature_vector["liwc:"+key] = liwc_scores[key]
  #
  # negative_score = liwc_scores["Negative Emotion"]
  # positive_score = liwc_scores["Positive Emotion"]
  #
  # if positive_score > negative_score:
  #     feature_vector["liwc:positive"] = 1
  # else:
  #     feature_vector["liwc:negative"] = 1
  texts = review['text'].to_dict()
  liwc_dict = {}
  for key, text in texts.items():
      liwc_scores = word_category_counter.score_text(text)
      liwc_dict[key] = liwc_scores
  review['liwc'] = pd.Series(get_frequencies(liwc_dict))


def getData():
  if os.path.exists(os.path.join(os.getcwd(), "jar_of_", "default_review.pkl")):
    review = pd.read_pickle(os.path.join(os.getcwd(), "jar_of_", "default_review.pkl"))
  else:
    review = pd.read_csv(os.path.join(os.getcwd(), "data", 'yelp_academic_dataset_review.csv'), encoding='utf-8')[['business_id', 'stars', 'text']].sample(
      frac=0.01, replace=False)
    review.to_pickle(os.path.join(os.getcwd(), "jar_of_", "default_review.pkl"))

  if os.path.exists(os.path.join(os.getcwd(), "jar_of_", "default_business.pkl")):
    business = pd.read_pickle(os.path.join(os.getcwd(), "jar_of_", "default_business.pkl"))
  else:
    business = pd.read_csv(os.path.join(os.getcwd(), "data", 'yelp_academic_dataset_business.csv'), low_memory=False, encoding='utf-8')[
    ['business_id', 'name', 'categories']]
    business.to_pickle(os.path.join(os.getcwd(), "jar_of_", "default_business.pkl"))
  categories = business['categories']
  star = review['stars']

  sentiment = []
  for s in star:
    if s <= 2:
      sentiment.append(0)
    else:
      sentiment.append(1)
  return review, business, categories, sentiment

def pickMaxCat():
  # pick the most possible category
  countCate = {}
  for c in categories:
    cateList = c.split(",")
    for cate in cateList:
      element = cate.strip(" '[]")
      countCate[element] = countCate.get(element, 0) + 1
  newCategories = []
  for c in categories:
    cateList = c.split(",")
    maxCate = 0
    finalCate = ""
    for cate in cateList:
      element = cate.strip(" '[]")
      if maxCate < countCate[element]:
        finalCate = element
        maxCate = countCate[element]
    newCategories.append(finalCate)
  business['categories'] = newCategories

def generateTfIdf(data):
  """
  Args:
      (dataFrame) data: whole data including review, business
  Returns:
      (dataFrame) data: whole data including review, business and tfIdf
  """
  def calculateTF(row):
    TFDict = {}
    for word in row:
      TFDict[word] = TFDict.get(word, 0) + 1
    TFDict = {k: float(v) / len(row) for k, v in TFDict.iteritems()}
    return TFDict
  IDFDict = {}
  def calculateIDF(row):
    row = set(row)
    for word in row:
      IDFDict[word] = IDFDict.get(word, 0) + 1
  def calculateTF_IDF(row):
    return {k: v * IDFDict[k] for k, v in row.items()}

  N = data.shape[0]
  TFDict = data['text'].apply(calculateTF)
  data['text'].apply(calculateIDF)
  IDFDict = {k: np.log(float(N) / v) for k, v in IDFDict.items()}
  TF_IDFDict = TFDict.apply(calculateTF_IDF).rename('TF_IDF')
  return pd.concat([data, TF_IDFDict], axis=1)

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
  return parser.parse_args()


if __name__ == "__main__":
  args = init()
  # Each argument is parsed as a boolean which defaults to False when not given.
  if args.unigram:
    print "unigram feature activated"
  if args.bigram:
    print "bigram feature activated"
  if args.liwc:
    print "LIWC feature activated"


  review, business, categories, sentiment = getData()

  # Feature: Generate unigram and bigram features if activated.
  # Also collect set of tokens in all reviews.
  unique_tokens = generate_ngram_feats(args.unigram, args.bigram, review)
  if args.liwc:
      add_liwc_features(review)
  if unique_tokens:
    sorted(list(unique_tokens))

  # Merge business and review DataFrames.
  mergeBusRev = pd.merge(business, review, on='business_id')
  featuresData = pd.concat([mergeBusRev, pd.DataFrame({'sentiment': sentiment})], axis=1)
  if not ( args.unigram or args.bigram ) and args.tfidf:
    print "Tf-Idf cannot be generated without unigram or bigram "
  elif args.tfidf:
    print "Tf-Idf feature activated"
    featuresData = generateTfIdf(featuresData)
  # Save the DataFrame with features to pickle.
  featuresData.to_pickle('jar_of_/features' + ''.join(sorted(sys.argv[1:])) + '.pkl')

