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

# Set of stopwords from NLTK
stops = set([str(w) for w in stopwords.words('english')])

def get_frequencies(review, unigram_activated, bigram_activated, tf_idf_activated):
  """Compute n-gram frequencies and add columns.
  :param review: DataFrame including tokens of each file
      (list) tokens for whole files @TODO what?
  Returns:
      (dataFrame) dataframe for frequency
  """
  def calculateTF(row):
    """
    Args:
        row: the feature entry in our set of features
    Returns:
        TF value
    """
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
    """
      Args:
          row: the feature entry in our set of features
      Returns:
          IDF value
      """
    unique_words = set(row['frequency'])
    for word in unique_words:
      IDFDict[word] = IDFDict.get(word, 0) + 1
  def calculateTF_IDF(row):
    """
    Args:
        row:  the feature entry in our set of features
    Returns:
        Combines the TF and IDF value
    """
    return {k: v * IDFDict[k] for k, v in row.items()}

  N = review.shape[0]
  review['frequency'] = review.apply(calculateTF, axis=1)

  #if we want tf_idf calculate, else use the baseline frequency calculation
  if tf_idf_activated:
    review.apply(calculateIDF, axis=1)
    IDFDict = {k: float(N) / v for k, v in IDFDict.items()}
    TF_IDFDict = review['frequency'].apply(calculateTF_IDF)
    review['frequency'] = TF_IDFDict
  return review

def generate_ngram_feats(unigram_activated, bigram_activated, tf_idf_activated, review):
  """ Generate the n-gram features that are activated.
  Add columns for unigram tokens and bigram tokens to the DataFrame review,
  and add the frequencies (unigram) and conditional frequencies (bigram).
  TF-IDF frequency is used if tf_idf_activated is True,
  otherwise regular frequency is used.
  Returns the modified review and .

  :type unigram_activated: boolean
  :type bigram_activated: boolean
  :type tf_idf_activated: boolean
  :type review: DataFrame
  :param review: 2-D DataFrame where rows are reviews, and columns include 'text'.
  """
  # Only perform the processing if at least one n-gram feature is activated.
  if not (unigram_activated or bigram_activated):
    return

  # Collect n-grams. We use NLTK sent_tokenize() for sentence tokenization.
  # We split words by spaces, lower-case all words, and remove all punctuation and stopwords.
  texts = review['text'].to_dict()
  unigram_dict = {}
  bigram_dict = {}
  count = 0
  size = len(texts.items())
  unique_uni = set()
  unique_bi  = set()
  for key, text in texts.items():
    count+=1
    # Print the progress to sys.stdout.
    if count % 1000 == 1:
      sys.stdout.write("%2.f" % (100.0 * count/size) + '%, completed: '+str(count)+'/'+str(size)+'\r')
      sys.stdout.flush()
    unigrams = []
    bigrams = []
    #sent tokenize
    for sent in nltk.sent_tokenize(text):
      tokens = []
      #word tokenize
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
  review = get_frequencies(review, unigram_activated, bigram_activated, tf_idf_activated)
  #word index, used in LR and SVM
  wordsIndex = {k: i for i, k in enumerate(unique_uni | unique_bi)}
  revWordsIndex = {i: k for i, k in enumerate(unique_uni | unique_bi)}

  return review, wordsIndex, revWordsIndex


def add_liwc_features(review):
  """
  Args:
      (dataframe)review

  Returns:
      the LIWC score of our file. Currently we say if Posemo > Negemo, the file is pos.

  """
  texts = review['text'].to_dict()
  liwc_dict = {}
  for key, text in texts.items():
    # All possible keys to liwc_scores start on line 269
    # of the word_category_counter.py script
      liwc_scores = word_category_counter.score_text(text)
      negative_score = liwc_scores["Negative Emotion"]
      positive_score = liwc_scores["Positive Emotion"]

      if positive_score > negative_score:
          liwc_dict[key] = {"liwc:positive":1}
      else:
          liwc_dict[key] = {"liwc:negative":1}
  #add liwc feature to existing feature
  def add_liwc_to_frequency(row):
      if 'frequency' in row:
        row['frequency'].update(row['liwc'])
      else:
        row['frequency'] = row['liwc']

  review['liwc'] = pd.Series(liwc_dict).reset_index(drop=True)
  review.reset_index(drop=True)
  review.apply(add_liwc_to_frequency, axis=1)
  return review


def getData():
  """
  Returns:
    The data parsed from the csv files broken into a few different variables:
    train_review = review frame for training data
    test_review = review frame for test data
    business = the business list from data
    categories = different categories of business
    train_sentiment = sentiment annotation for train set
    test_sentiment = sentiment annotation for test set
  """
  #datafiles are stored as pickles to make load time not so expensive during dev
  if os.path.exists(os.path.join(os.getcwd(), "jar_of_", "default_train_review.pkl")):
    train_review = pd.read_pickle(os.path.join(os.getcwd(), "jar_of_", "default_train_review.pkl"))
  else:
    train_review = pd.read_csv(os.path.join(os.getcwd(), "data", 'yelp_academic_dataset_review.csv'), encoding='utf-8')[['business_id', 'stars', 'text']].sample(
      frac=.1, replace=False)
    train_review.to_pickle(os.path.join(os.getcwd(), "jar_of_", "default_train_review.pkl"))

  if os.path.exists(os.path.join(os.getcwd(), "jar_of_", "default_test_review.pkl")):
    test_review = pd.read_pickle(os.path.join(os.getcwd(), "jar_of_", "default_test_review.pkl"))
  else:
    test_review = pd.read_csv(os.path.join(os.getcwd(), "data", 'yelp_academic_dataset_review.csv'), encoding='utf-8')[['business_id', 'stars', 'text']].sample(
      frac=.02, replace=False)
    test_review.to_pickle(os.path.join(os.getcwd(), "jar_of_", "default_test_review.pkl"))

  if os.path.exists(os.path.join(os.getcwd(), "jar_of_", "default_business.pkl")):
    business = pd.read_pickle(os.path.join(os.getcwd(), "jar_of_", "default_business.pkl"))
  else:
    business = pd.read_csv(os.path.join(os.getcwd(), "data", 'yelp_academic_dataset_business.csv'), low_memory=False, encoding='utf-8')[
    ['business_id', 'name', 'categories']]
    business.to_pickle(os.path.join(os.getcwd(), "jar_of_", "default_business.pkl"))
  categories = business['categories']
  train_star = train_review['stars']

  #sentiment classification is handled automatically using the stars. star rating of 0,1,2 is neg, 3,4,5 is pos
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



def get_args():
  """Parse command line arguments.
  Each argument is parsed as a boolean which defaults to False when not given.
  """

  parser = argparse.ArgumentParser(description="Specify feature types")
  parser.add_argument("-u", "--unigram", help="activate the unigram feature",
                      action="store_true")
  parser.add_argument("-b", "--bigram", help="activate the bigram feature",
                      action="store_true")
  parser.add_argument("-l", "--liwc", help="activate the LIWC feature",
                      action="store_true")
  parser.add_argument("-a", "--all", help="create maximum combos and pickle each",
                      action="store_true")
  parser.add_argument("-t", "--tfidf", help="use TF-IDF frequency count",
                      action="store_true")

  args = parser.parse_args()
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
  return args

if __name__ == "__main__":
  args = get_args()

  train_review, test_review, business, categories, train_sentiment, test_sentiment = getData()
  if not os.path.exists(os.path.join(os.getcwd(), "jar_of_", "pickle" + ''.join(sorted(sys.argv[1:])))):
      os.makedirs(os.path.join(os.getcwd(), "jar_of_", "pickle" + ''.join(sorted(sys.argv[1:]))))
  # Feature: Generate unigram and bigram features if activated.
  # Also collect set of tokens in all reviews.
  train = generate_ngram_feats(args.unigram, args.bigram, args.tfidf, train_review)
  if train != None:
    train_review = train[0]
    #liwc data
    if args.liwc:
        wordsIndexSize = len(train[1])
        train[1].update({'liwc:positive': wordsIndexSize, 'liwc:negative': wordsIndexSize + 1})
        train[2].update({wordsIndexSize : 'liwc:positive', wordsIndexSize + 1 : 'liwc:negative'})
        test_review = add_liwc_features(test_review)

    #Store word index pickles for use in SVM or LR
    pickle.dump(train[1], open('jar_of_/pickle'+''.join(sorted(sys.argv[1:]))+'/wordsIndex.pkl', 'wb'))
    pickle.dump(train[2], open('jar_of_/pickle' + ''.join(sorted(sys.argv[1:])) + '/revWordsIndex.pkl', 'wb'))
  test = generate_ngram_feats(args.unigram, args.bigram, args.tfidf, test_review)
  if test != None:
      test_review = test[0]

  # Merge business and review DataFrames.
  mergeBusRev = pd.merge(business, train_review, on='business_id')
  featuresData = pd.concat([mergeBusRev, pd.DataFrame({'sentiment': train_sentiment})], axis=1)
  # Save the DataFrame with features to pickle.
  featuresData.to_pickle('jar_of_/pickle'+''.join(sorted(sys.argv[1:])) + '/train_features.pkl')

  mergeBusRev = pd.merge(business, test_review, on='business_id')
  featuresData = pd.concat([mergeBusRev, pd.DataFrame({'sentiment': test_sentiment})], axis=1)
  # Save the DataFrame with features to pickle.
  featuresData.to_pickle('jar_of_/pickle'+''.join(sorted(sys.argv[1:])) + '/test_features.pkl')
