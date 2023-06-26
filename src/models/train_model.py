"""Trains a model to predict the sentiment of a restaurant review"""
import time
import logging
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, f1_score
import click


@click.command()
@click.argument(
    "dataset_filepath",
    type=click.Path(exists=True),
    default="./data/processed/restaurant_reviews.csv",
)
@click.argument(
    "output_filepath", type=click.Path(), default="./models/naive_bayes_classifier.pkl"
)
@click.argument("metrics_filepath", type=click.Path(), default="./models/metrics.json")
@click.argument("random_state", type=int, default=0)
# Disabling too many locals warning, as all local variables are necessary
# pylint: disable=too-many-locals
def main(dataset_filepath, output_filepath, metrics_filepath, random_state=0):
    """Trains a model to predict the sentiment of a restaurant review"""
    logger = logging.getLogger(__name__)
    logger.info("loading processed dataset from %s", dataset_filepath)

    # Loading dataset (no column selection necessary)
    df: pd.DataFrame = pd.read_csv(
        dataset_filepath, dtype={"Review": str, "Liked": int}
    )
    df = df.dropna()

    # Data transformation (pylint doesn't recognize iloc, so we disable the warning)
    # pylint: disable=no-member
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1].values

    # Dividing dataset into training and test set
    logger.info(
        "dividing dataset into training and test set,\
                 with test size of 20%"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )

    logger.info("training model")

    start = time.time()
    model = train(X_train, y_train)
    stop = time.time()

    y_pred = model.predict(X_test)

    metrics = {
        "train": {
            "accuracy": model.score(X_train, y_train),
            "roc_auc": roc_auc_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "train_size": len(X_train),
            "train_time": stop - start,
        },
        "test": {
            "accuracy": model.score(X_test, y_test),
            "roc_auc": roc_auc_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "test_size": len(X_test),
        },
    }

    logger.info("model accuracy on test set: %.2f%%", model.score(X_test, y_test) * 100)

    # Save the metric to json for DVC
    with open(metrics_filepath, "w", encoding="utf-8") as json_file:
        json.dump(metrics, json_file)

    # Exporting the classifier pipeline to later use in prediction
    logger.info("saving model to %s", output_filepath)

    joblib.dump(model, output_filepath)


def train(X, y):
    """Trains a model to predict the sentiment of a restaurant review"""

    # Model fitting (Naive Bayes)
    column_transformer = ColumnTransformer(
        [("vectorizer", CountVectorizer(max_features=1420), "Review")]
    )

    model = Pipeline(
        [("transform", column_transformer), ("classifier", MultinomialNB())]
    )

    model.fit(X, y)

    return model


if __name__ == "__main__":
    # Ignore pylint error for click decorated methods
    # pylint: disable=no-value-for-parameter
    main()
