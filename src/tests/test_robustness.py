from typing import Generator
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.train_model import train
import pytest


class TestRobustness:
    @pytest.fixture
    def df(self) -> Generator[pd.DataFrame, None, None]:
        df = pd.read_csv(
            "data/processed/restaurant_reviews.csv", dtype={"Review": str, "Liked": int}
        )
        yield df

    def evaluate_score(self, model, X_test, y_test):
        return model.score(X_test, y_test)

    def test_non_determinism_robustness(self, df: pd.DataFrame):
        X, y = df.iloc[:, 0:-1], df.iloc[:, -1].values
        scores: list[float] = []
        for seed in range(5):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=seed
            )
            model = train(X_train, y_train)
            new_score = self.evaluate_score(model, X_test, y_test)
            for score in scores:
                # TODO: threshold at 0.1 is too high... 10% difference in score is too much
                assert score == pytest.approx(new_score, 0.1)
            scores.append(new_score)
