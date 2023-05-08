import pandas as pd
import pickle
import joblib

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

from text_preprocessing import prepareStopwords, dataPreprocess, dataTransform


if __name__ == '__main__':
   # Loading dataset
   dataset = pd.read_csv('a1_RestaurantReviews_HistoricDump.tsv', delimiter = '\t', quoting = 3)

   # Data preprocessing
   ps, all_stopwords = prepareStopwords()
   corpus=[]
   for i in range(0, 900):
      review = dataPreprocess(dataset['Review'][i], ps, all_stopwords)
      corpus.append(review)

   # Data transformation
   X, cv = dataTransform(corpus)
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
   acc = accuracy_score(y_test, y_pred)
   print(acc)

   # Exporting NB Classifier to later use in prediction
   joblib.dump(classifier, 'c2_Classifier_Sentiment_Model') 


