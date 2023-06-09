from typing import Generator
import pandas as pd
import pytest
import joblib


class TestCapabilities:
    @pytest.fixture
    def trained_model(self):
        return joblib.load("models/naive_bayes_classifier.pkl")

    @pytest.fixture
    def test_data(self) -> Generator[pd.DataFrame, None, None]:
        df = pd.read_csv(
            "data/processed/restaurant_reviews.csv", dtype={"Review": str, "Liked": int}
        )
        yield df.dropna()

    def evaluate_score(self, model, X_test, y_test):
        return model.score(X_test, y_test)

    def test_model_on_food_data_slice(self, trained_model, test_data: pd.DataFrame):
        X, y = test_data.iloc[:, 0:-1], test_data.iloc[:, -1].values

        normal_score = self.evaluate_score(trained_model, X, y)

        # slicing based on the word food
        sliced_data = test_data[test_data["Review"].str.contains("food", case=False)]

        sliced_X, sliced_y = sliced_data.iloc[:, 0:-1], sliced_data.iloc[:, -1].values

        sliced_score = self.evaluate_score(trained_model, sliced_X, sliced_y)

        assert normal_score == pytest.approx(sliced_score, 0.2)
