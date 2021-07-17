
#Importing libraries
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score


# Reading from the input file
train = pd.read_csv('train_data_labels.csv')

# Normalizing each of the abstract entries
for entry in train['abstract']:
    for each in entry:
        each = each.lower()

# Tokenizing each of the abstract entries
toks = []
for entry in train['abstract']:
    toks.append(word_tokenize(entry))
train['abstract_mod'] = toks
train = train.drop('abstract',axis=1)
train['abstract'] = train['abstract_mod']
train = train.drop('abstract_mod',axis=1)

# Using Lemmatisation and POS-tagging
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


for index,entry in enumerate(train['abstract']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    train.loc[index,'abstract_final'] = str(Final_words)

# Splitting the data into train and test sets
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(train['abstract_final'],train['label'],test_size=0.3)

# Using LabelEncoder and TfidfVectorizer for feature extraction
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(train['abstract_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# Initialising and fitting the SVM model
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)


# Preprocessing the abstracts and predicting the class of each abstracts using the SVM model
test = pd.read_csv("test_data.csv")

test = pd.read_csv("test_data.csv")
for entry in test['abstract']:
    for each in entry:
        each = each.lower()
toks = []
for entry in test['abstract']:
    toks.append(word_tokenize(entry))
test['abstract_mod'] = toks
test = test.drop('abstract',axis=1)
test['abstract'] = test['abstract_mod']
test = test.drop('abstract_mod',axis=1)

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


for index,entry in enumerate(test['abstract']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    test.loc[index,'abstract_final'] = str(Final_words)

Test_X_Tfidf2 = Tfidf_vect.transform(test['abstract_final'])

predictions_test_data = SVM.predict(Test_X_Tfidf2)

predictions_test_data_final = [each for each in predictions_test_data]
test['labels_numeric'] = predictions_test_data_final

train_copy=train

num_vals = []
num_vals_temp = Encoder.fit_transform(train_copy['label'])
for each in num_vals_temp:
    num_vals.append(each)

train_copy['num_labels'] = num_vals

vals_to_lab = []
for i in range(len(train_copy)):
    vals_to_lab.append((train_copy['label'].iloc[i],train_copy['num_labels'].iloc[i]))
vals_to_lab = list(set(vals_to_lab))

test_labels_vals = []
for i in range(len(test)):
    for each in vals_to_lab:
        if test['labels_numeric'].iloc[i] == each[1]:
            test_labels_vals.append(each[0])
            
            
test['label'] = test_labels_vals
test = test.drop('abstract',axis = 1)
test = test.drop('abstract_final',axis = 1)
test = test.drop('labels_numeric',axis=1)
test.to_csv('pred_labels_two.csv',index=False)
# predictions_test = Encoder.inverse_transform(predictions_SVM)

# test['label']=predictions_test
# test.drop(['abstract','abstract_final'],inplace=True,axis=1)
# test.to_csv("pred_labels.csv",index=False)






