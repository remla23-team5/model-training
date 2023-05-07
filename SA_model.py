import numpy as np
import pandas as pd
import re
import pickle
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score


# Loading dataset
dataset = pd.read_csv('a1_RestaurantReviews_HistoricDump.tsv', delimiter = '\t', quoting = 3)


# Data preprocessing
nltk.download('stopwords')
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

corpus=[]

for i in range(0, 900):
   review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
   review = review.lower()
   review = review.split()
   review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
   review = ' '.join(review)
   corpus.append(review)


# Data transformation
cv = CountVectorizer(max_features = 1420)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values


# Saving BoW dictionary to later use in prediction
bow_path = 'c1_BoW_Sentiment_Model.pkl'
pickle.dump(cv, open(bow_path, "wb"))


# Dividing dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Model fitting (Naive Bayes)
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Model performance
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc = accuracy_score(y_test, y_pred)
print(acc)


# Exporting NB Classifier to later use in prediction
joblib.dump(classifier, 'c2_Classifier_Sentiment_Model') 


