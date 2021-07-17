# Document_Classification
Machine learning model to classify documents using python


This package contains the  source code and binaries for 
model described in:

The directory contents of this distribution are as follows:

* Document_classification.py        - python 3
* README          		          - This documentation

And a zip file data which contains

* test_data   			          - Test data
* train_data_labels		          - Train data
* pred_labels_two              - The final output

============================================================

Running Instruction

Use jupyter notebook to run the code in Python 3.

Library used: 
* import pandas as pd
* import numpy as np
* from nltk.tokenize import word_tokenize
* from nltk import pos_tag
* from nltk.corpus import stopwords
* from nltk.stem import WordNetLemmatizer
* from sklearn.preprocessing import LabelEncoder
* from collections import defaultdict
* from nltk.corpus import wordnet as wn
* from sklearn.feature_extraction.text import TfidfVectorizer
* from sklearn import model_selection, naive_bayes, svm
* from sklearn.metrics import accuracy_score

============================================================

Input data format

The format of input data is as requirement. There are two files

train_data_labels          - includes id, texts, labels
test_data                 - includes id, texts

============================================================

Output files
pred_labels_two.csv            - includes id and labels
