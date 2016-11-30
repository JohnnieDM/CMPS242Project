# CMPS242Project
This is the class project for CMPS 242 using data from the Yelp Dataset Challenge.


## Preprocessing
Tools we need:
- pandas
```
sudo pip install pandas
```
- nltk
```
sudo pip install nltk
```
- download packages in nltk
```
>>> import nltk
>>> nltk.download()
```
- scikit-learn for comparison
```
sudo pip install -U scikit-learn
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
Put these two files into 'data' directory.
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
Then find all the features file will be put into the directory
```
jar_of_/pickle-l-t-u
```

##Modeling and Prediction
Run the file Process.py with required keyword arguments (-c, -d) to train a model and predict.
Run the following command to detailed messages about the required arguments.
```
python Process.py -h
```
For instance, run the command below to build a Naive Bayes classifier trained on the features selected above (unigrams, LIWC scores, and TF-IDF frequency).
```
python Process.py -c nb -d jar_of_/pickle-l-t-u
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
