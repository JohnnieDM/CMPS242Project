# CMPS242Project
This is the class project for CMPS 242 using data from the Yelp Dataset Challenge.


## Preprocessing
Tools we need:
- pandas
```
sudo pip install pandas
```
Data we need:
- yelp_academic_dataset_business.json
- yelp_academic_dataset_review.json

First use json_to_csv_converter.py from this [repository](https://github.com/Yelp/dataset-examples) to convert the json files into csv format (yelp_academic_dataset_business.json and yelp_academic_dataset_review.json to yelp_academic_dataset_business.csv and yelp_academic_dataset_review.csv).

Use these two commands on the shell.
```
python json_to_csv_converter.py yelp_academic_dataset_business.json
python json_to_csv_converter.py yelp_academic_dataset_review.json
```
Then run Preprocess.py to generate pickled feature files.
Here we randomly sample 1% of the dataset, since processing the entire dataset would take too much time.
Optionally give flags (-u, -b, -l, -a, -t) to select the features to use.
Run the following command to see detailed messages about the options.
```
python Preprocess.py -h
```
For instance, run the command below to generate feature files using unigrams, LIWC scores, and TF-IDF frequency:
```
python Preprocess.py -u -l -t
```
Then find the following feature files:
```
jar_of_/train_features-l-t-u.pkl
jar_of_/train_revWordsIndex-l-t-u.pkl
jar_of_/train_wordsIndex-l-t-u.pkl
jar_of_/test_features-l-t-u.pkl
```

##Modeling and Prediction
Run the file Process.py with required keyword arguments (-c, -t, -s, -w, -r) to train a model and predict.
Run the following command to detailed messages about the required arguments.
```
python Process.py -h
```
For instance, run the command below to build a Naive Bayes classifier trained on the features selected above (unigrams, LIWC scores, and TF-IDF frequency).
```
python Process.py -c nb -t jar_of_/train_features-l-t-u.pkl -s jar_of_/test_features-l-t-u.pkl -w jar_of_/train_wordsIndex-l-t-u.pkl -r jar_of_/train_revWordsIndex-l-t-u.pkl
```
The prediction result will be printed to sys.stdout as follows, showing the accuracy, precision, recall, and F1 score:
```
=====================================================
Results:
  Accuracy:     0.752411455812
  Precision:    0.793605698051
  Recall:       0.930122403039
  F1 score:     0.856458090426
=====================================================
```
