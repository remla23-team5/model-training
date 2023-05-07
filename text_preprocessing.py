import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

def prepareStopwords():
    nltk.download('stopwords')
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    return ps, all_stopwords


def dataPreprocess(review, ps, all_stopwords):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review

def dataTransform(corpus):
    cv = CountVectorizer(max_features = 1420)
    X = cv.fit_transform(corpus).toarray()
    return X, cv