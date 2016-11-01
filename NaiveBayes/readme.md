# Naive Bayes
All our work is based on this [article](http://rstudio-pubs-static.s3.amazonaws.com/155272_c91116f9fd774349bef82cb154b62f5c.html)
## Preprocess
Data we need:  
- yelp_academic_dataset_business.json
- yelp_academic_dataset_review.json
First, we need to use json_to_csv_converter.py from this [repository](https://github.com/Yelp/dataset-examples) to convert yelp_academic_dataset_business.json and yelp_academic_dataset_review.json to yelp_academic_dataset_business.csv and yelp_academic_dataset_review.csv.  
Use these two command on the shell  
```
python json_to_csv_converter.py yelp_academic_dataset_business.json
python json_to_csv_converter.py yelp_academic_dataset_review.json
```
Then we can run the file we make: NaiveBayesDataPreprocess.py to output a data file: NaiveBayesData.csv.  
The attributes in this file are:  
- business_id
- name
- categories
- stars
- text
- sentiment

