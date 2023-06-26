from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import pandas as pd
from typing import Generator
import pytest


class TestFeatureDistribution():
    @pytest.fixture
    def df(self) -> Generator[pd.DataFrame, None, None]:
        df = pd.read_csv(
            "data/processed/restaurant_reviews.csv", dtype={"Review": str, "Liked": int}
        )
        yield df.dropna()

    def test_monitoring_invariants(self, df: pd.DataFrame):
        """
        Test that the distribution of features in the training and test set are similar using the P.S. test.
        """
        X, y = df.iloc[:, 0:-1], df.iloc[:, -1].values
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        for feature in X_train.columns:
            feature_sample_training = X_train.loc[:, feature]
            feature_sample_test = X_test.loc[:, feature]

            p_value = ks_2samp(feature_sample_training, feature_sample_test).pvalue

            assert p_value > 0.05
