"""
This module reads raw data and preprocesses it to be used for model training.
"""

# -*- coding: utf-8 -*-
import logging

import pandas as pd

import click

from lib2.preprocessing import prepare_stopwords, preprocess_data
from lib2.version_util import VersionUtil


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data at %s", input_filepath)

    # Loading dataset (no column selection necessary)
    dataset = pd.read_csv(
        input_filepath, delimiter="\t", quoting=3, dtype={"Review": str, "Liked": int}
    )

    # Data preprocessing
    logger.info(
        "preprocessing data, using lib version: %s", VersionUtil().get_version()
    )

    stemmer, english_stopwords = prepare_stopwords()

    dataset["Review"] = dataset["Review"].apply(
        preprocess_data, args=(stemmer, english_stopwords)
    )

    logger.info("outputting processed data set to %s", output_filepath)

    dataset.to_csv(output_filepath, index=False)

    logger.info("done")


if __name__ == "__main__":
    # Ignore pylint error for click decorated methods
    # pylint: disable=no-value-for-parameter
    main()
