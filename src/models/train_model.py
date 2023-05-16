import pandas as pd
import joblib
import click
import logging

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


@click.command()
@click.argument('dataset_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(dataset_filepath, output_filepath):
    """ Trains a model to predict the sentiment of a restaurant review
    """
    logger = logging.getLogger(__name__)
    logger.info(f'loading processed dataset from {dataset_filepath}')

    # Loading dataset
    df = pd.read_csv(dataset_filepath)
    df = df.dropna()

    # Data transformation
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1].values

    # Dividing dataset into training and test set
    logger.info('dividing dataset into training and test set, with test size of 20%')
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

    logger.info(f'model accuracy on test set: {model.score(X_test, y_test)}')

    # Exporting the classifier pipeline to later use in prediction
    logger.info(f'saving model to {output_filepath}')

    joblib.dump(model, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
