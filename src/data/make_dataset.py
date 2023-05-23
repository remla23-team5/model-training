"""
This module reads raw data and preprocesses it to be used for model training.
"""

# -*- coding: utf-8 -*-
import logging
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd

import click


def prepare_stopwords():
    """Download stopwords and create stemmer for data preprocessing."""
    nltk.download('stopwords')
    stemmer = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    return stemmer, all_stopwords


def preprocess_data(review, stemmer, words_to_remove):
    """Preprocess data by removing stopwords and punctuation,
       and lower-casing + stemming the words.
    """
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower().split()
    review = [stemmer.stem(w) for w in review if w not in set(words_to_remove)]
    review = ' '.join(review)
    return review


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data at %s', input_filepath)

    # Loading dataset
    # pylint: disable=column-selection-pandas
    dataset = pd.read_csv(
        input_filepath,
        delimiter='\t',
        quoting=3,
        dtype={'Review': str, 'Liked': int}
    )

    # Data preprocessing
    stemmer, english_stopwords = prepare_stopwords()

    dataset['Review'] = dataset['Review']\
        .apply(preprocess_data, args=(stemmer, english_stopwords))

    logger.info('outputting processed data set to %s', output_filepath)

    dataset.to_csv(output_filepath, index=False)

    logger.info('done')


if __name__ == '__main__':
    # Ignore pylint error for click decorated methods
    # pylint: disable=no-value-for-parameter
    main()
