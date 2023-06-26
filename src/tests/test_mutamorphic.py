import pytest
import joblib
import nlpaug.augmenter.word as naw
from typing import Generator
import pandas as pd
import numpy as np
import time


class TestMutamorphic:
    @pytest.fixture
    def trained_model(self):
        return joblib.load("models/naive_bayes_classifier.pkl")

    @pytest.fixture
    def test_data(self) -> Generator[pd.DataFrame, None, None]:
        test_data = pd.read_csv(
            "data/processed/restaurant_reviews.csv", dtype={"Review": str, "Liked": int}
        )
        yield test_data.dropna()

    def test_word_mutations(self, trained_model, test_data):
        aug = naw.SynonymAug(aug_src="wordnet")

        X, _ = test_data.iloc[:, 0:-1], test_data.iloc[:, -1].values

        X_aug = []
        for origin in X["Review"]:
            mutant = str(aug.augment(origin))
            X_aug.append(mutant)
        X_mutant = pd.DataFrame({"Review": X_aug})

        y_pred = trained_model.predict(X)
        y_pred_mutated = trained_model.predict(X_mutant)

        idxs = np.where(y_pred != y_pred_mutated)[0]
        no_mutant = 0

        for idx in idxs:
            origin = str(X["Review"].iloc[idx])
            mutant = aug.augment(origin)
            mutant_pd = pd.DataFrame({"Review": mutant})

            y_pred_ori = y_pred[idx]
            y_pred_per = trained_model.predict(mutant_pd)[0]

            start_time = time.time()
            while y_pred_ori != y_pred_per:
                mutant = aug.augment(origin)
                mutant_pd = pd.DataFrame({"Review": mutant})
                y_pred_per = trained_model.predict(mutant_pd)[0]
                if time.time() > start_time + 2:
                    no_mutant += 1
                    break

        assert no_mutant < 10
