"""Trains a model to predict the sentiment of a restaurant review"""

import logging
import joblib
import json
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import click


@click.command()
@click.argument('dataset_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(dataset_filepath, output_filepath):
    """ Trains a model to predict the sentiment of a restaurant review
    """
    logger = logging.getLogger(__name__)
    logger.info('loading processed dataset from %s', dataset_filepath)

    # Loading dataset (no column selection necessary)
    df = pd.read_csv(dataset_filepath, dtype={'Review': str, 'Liked': int})
    df = df.dropna()

    # Data transformation
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1].values

    # Dividing dataset into training and test set
    logger.info('dividing dataset into training and test set,\
                 with test size of 20%')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Model fitting (Naive Bayes)
    column_transformer = ColumnTransformer([
        ('vectorizer', CountVectorizer(max_features=1420), 'Review')
    ])

    model = Pipeline([
            ('transform', column_transformer),
            ('classifier', MultinomialNB())
    ])

    logger.info('fitting model')
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    accuracy_json = {"accuracy": accuracy}

    logger.info('model accuracy on test set: %.2f%%', accuracy * 100)

    # Exporting the classifier pipeline to later use in prediction
    logger.info('saving model to %s', output_filepath)

    joblib.dump(model, output_filepath)

    # Save the metric to json for DVC
    json_path = "./models/accuracy.json"
    with open(json_path, "w") as f:
        json.dump(accuracy_json, f)

if __name__ == '__main__':
    # Ignore pylint error for click decorated methods
    # pylint: disable=no-value-for-parameter
    main()
