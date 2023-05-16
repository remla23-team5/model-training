# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd

from pathlib import Path

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def prepare_stopwords():
    nltk.download('stopwords')
    stemmer = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    return stemmer, all_stopwords


def preprocess_data(review, stemmer, all_stopwords):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [stemmer.stem(word) for word in review if word not in set(all_stopwords)]
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
    logger.info(f'making final data set from raw data at {input_filepath}')

    # Loading dataset
    dataset = pd.read_csv(input_filepath, delimiter='\t', quoting=3)

    # Data preprocessing
    stemmer, all_stopwords = prepare_stopwords()

    dataset['Review'] = dataset['Review'].apply(preprocess_data, args=(stemmer, all_stopwords))

    logger.info(f'outputting processed data set to {output_filepath}')

    dataset.to_csv(output_filepath, index=False)

    logger.info('done')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
