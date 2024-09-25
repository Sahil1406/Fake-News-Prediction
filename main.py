# IMPORTING ALL THE DEPENDENCIES

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# print(stopwords.words('English'))

dataset = pd.read_csv('train.csv')

# CHECKING OUT THE DATA

# print(dataset.shape)
# print(dataset.head())
# print(dataset.isnull().sum())

# REPLACING NULL VALUES WITH EMPTY STRING

dataset = dataset.fillna('')

# MERGING THE AUTHOR TITLE AND TEXT TOGETHER

dataset['content'] = dataset['author']+' '+ dataset['title']

# print(dataset['content'])

# SEPARATING THE USEFUL DATA AND LABELS

X = dataset.drop('label', axis=1)
Y = dataset['label']

# print(X)
# print(Y)

# STEMMING 
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content) 
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

dataset['content'] = dataset['content'].apply(stemming)

# print('done')

# SEPARATING THE DATA AND THE LABEL

X = dataset['content'].values

Y = dataset['label'].values

print(X.shape, Y.shape)


# CONVERTING TEXTUAL DATA TO NUMERICAL DATA

vectorizer = TfidfVectorizer()

vectorizer.fit(X)

X = vectorizer.transform(X)


# SPITTING THE DATA INTO TRAINING AND TEST DATA

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.8, stratify = Y, random_state=1)


# TRAINING THE MODEL

model = LogisticRegression()

model.fit(X_train, Y_train)

# EVALUATION


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on training data is: ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on testing data is: ', testing_data_accuracy)