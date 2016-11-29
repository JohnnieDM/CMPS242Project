import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
import sys
import argparse
import word_category_counter
import os
import pickle

stops = set([str(w) for w in stopwords.words('english')])

def get_frequencies(review, unigram_activated, bigram_activated, tf_idf_activated):
  """
  Args:
      (dataFrame) df: review dataframe including tokens of each file
      (list) tokens for whole files
  Returns:
      (dataFrame) dataframe for frequency
  """
  def calculateTF(row):
    uni_tokens = row['uni_tokens']
    TFDict = {}
    for word in uni_tokens:
      TFDict[word] = TFDict.get(word, 0) + 1
    TFDict = {k: float(v) / len(uni_tokens) for k, v in TFDict.iteritems()}
    if bigram_activated:
        bi_tokens = row['bi_tokens']
        tempDict = {}
        for words in bi_tokens:
            tempDict[words] = tempDict.get(word, 0) + 1
        tempDict = {k: float(v) / (len(bi_tokens)) for k, v in tempDict.iteritems()}
        if unigram_activated:
          TFDict.update(tempDict)
        else:
          TFDict = tempDict
    return TFDict
  IDFDict = {}
  def calculateIDF(row):
    unique_words = set(row['frequency'])
    for word in unique_words:
      IDFDict[word] = IDFDict.get(word, 0) + 1
  def calculateTF_IDF(row):
    return {k: v * IDFDict[k] for k, v in row.items()}

  N = review.shape[0]
  review['frequency'] = review.apply(calculateTF, axis=1)

  if tf_idf_activated:
    review.apply(calculateIDF, axis=1)
    IDFDict = {k: float(N) / v for k, v in IDFDict.items()}
    TF_IDFDict = review['frequency'].apply(calculateTF_IDF)
    review['frequency'] = TF_IDFDict
  return review

def generate_ngram_feats(unigram_activated, bigram_activated, tf_idf_activated, review):
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
        bigrams.extend(list(ngrams(tokens,2))) # list() to unpack the bigram generator object

    unigram_dict[key] = unigrams
    bigram_dict[key] = bigrams
    [unique_uni.add(unigram) for unigram in unigrams]
    [unique_bi.add(bigram) for bigram in bigrams]

  # Add columns of n-gram tokens of each text.
  review['uni_tokens'] = pd.Series(unigram_dict)
  if bigram_activated:
    review['bi_tokens'] = pd.Series(bigram_dict)
  review = review.reset_index(drop=True)

  # Add columns of frequencies.
  # Turn sets of unique tokens into sorted lists for frequency computation.
  review = get_frequencies(review, unigram_activated, bigram_activated, tf_idf_activated)
  #print review
  wordsIndex = {k: i for i, k in enumerate(unique_uni | unique_bi)}
  revWordsIndex = {i: k for i, k in enumerate(unique_uni | unique_bi)}

  return wordsIndex, revWordsIndex


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
      # liwc_dict[key] = liwc_scores
      negative_score = liwc_scores["Negative Emotion"]
      positive_score = liwc_scores["Positive Emotion"]

      if positive_score > negative_score:
          liwc_dict[key] = {"liwc:positive":1}
      else:
          liwc_dict[key] = {"liwc:negative":1}

  review['liwc'] = pd.Series(liwc_dict)
  #print review


def getData():
  if os.path.exists(os.path.join(os.getcwd(), "jar_of_", "default_train_review.pkl")):
    train_review = pd.read_pickle(os.path.join(os.getcwd(), "jar_of_", "default_train_review.pkl"))
  else:
    train_review = pd.read_csv(os.path.join(os.getcwd(), "data", 'yelp_academic_dataset_review.csv'), encoding='utf-8')[['business_id', 'stars', 'text']].sample(
      frac=0.01, replace=False)
    train_review.to_pickle(os.path.join(os.getcwd(), "jar_of_", "default_train_review.pkl"))

  if os.path.exists(os.path.join(os.getcwd(), "jar_of_", "default_test_review.pkl")):
    test_review = pd.read_pickle(os.path.join(os.getcwd(), "jar_of_", "default_test_review.pkl"))
  else:
    test_review = pd.read_csv(os.path.join(os.getcwd(), "data", 'yelp_academic_dataset_review.csv'), encoding='utf-8')[['business_id', 'stars', 'text']].sample(
      frac=0.01, replace=False)
    test_review.to_pickle(os.path.join(os.getcwd(), "jar_of_", "default_test_review.pkl"))

  if os.path.exists(os.path.join(os.getcwd(), "jar_of_", "default_business.pkl")):
    business = pd.read_pickle(os.path.join(os.getcwd(), "jar_of_", "default_business.pkl"))
  else:
    business = pd.read_csv(os.path.join(os.getcwd(), "data", 'yelp_academic_dataset_business.csv'), low_memory=False, encoding='utf-8')[
    ['business_id', 'name', 'categories']]
    business.to_pickle(os.path.join(os.getcwd(), "jar_of_", "default_business.pkl"))
  categories = business['categories']
  train_star = train_review['stars']

  train_sentiment = []
  for s in train_star:
    if s <= 2:
      train_sentiment.append(0)
    else:
      train_sentiment.append(1)
  test_star = test_review['stars']

  test_sentiment = []
  for s in test_star:
    if s <= 2:
      test_sentiment.append(0)
    else:
      test_sentiment.append(1)
  return train_review, test_review, business, categories, train_sentiment, test_sentiment

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
  if args.all:
      args.unigram, args.bigram, args.liwc, args.tfidf = True, True, True, True
  if args.unigram:
    print "unigram feature activated"
  if args.bigram:
    print "bigram feature activated"
  if args.liwc:
    print "LIWC feature activated"
  if args.tfidf:
    print "Tf-Idf feature activated"

  train_review, test_review, business, categories, train_sentiment, test_sentiment = getData()

  # Feature: Generate unigram and bigram features if activated.
  # Also collect set of tokens in all reviews.
  train_index = generate_ngram_feats(args.unigram, args.bigram, args.tfidf, train_review)
  if train_index != None:
    pickle.dump(train_index[0], open('jar_of_/train_wordsIndex'+''.join(sorted(sys.argv[1:]))+'.pkl', 'wb'))
    pickle.dump(train_index[1], open('jar_of_/train_revWordsIndex' + ''.join(sorted(sys.argv[1:])) + '.pkl', 'wb'))
  generate_ngram_feats(args.unigram, args.bigram, args.tfidf, test_review)
  if args.liwc:
      add_liwc_features(train_review)



  # Merge business and review DataFrames.
  mergeBusRev = pd.merge(business, train_review, on='business_id')
  featuresData = pd.concat([mergeBusRev, pd.DataFrame({'sentiment': train_sentiment})], axis=1)
  # Save the DataFrame with features to pickle.
  featuresData.to_pickle('jar_of_/train_features' + ''.join(sorted(sys.argv[1:])) + '.pkl')

  mergeBusRev = pd.merge(business, test_review, on='business_id')
  featuresData = pd.concat([mergeBusRev, pd.DataFrame({'sentiment': test_sentiment})], axis=1)
  # Save the DataFrame with features to pickle.
  featuresData.to_pickle('jar_of_/test_features' + ''.join(sorted(sys.argv[1:])) + '.pkl')
