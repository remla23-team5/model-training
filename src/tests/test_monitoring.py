import pytest

import psutil
from typing import Generator
import pandas as pd
import time

from src.models.train_model import train


class TestMonitoring():
    @pytest.fixture
    def df(self) -> Generator[pd.DataFrame, None, None]:
        df = pd.read_csv(
            "data/processed/restaurant_reviews.csv", dtype={"Review": str, "Liked": int}
        )
        yield df.dropna()

    def test_memory_usage_during_training(self, df) -> None:
        X, y = df.iloc[:, 0:-1], df.iloc[:, -1].values

        process = psutil.Process()
        start_ram = process.memory_info().rss  # in bytes
        _ = train(X, y)

        # test for training ram
        end_ram = process.memory_info().rss
        ram_used = end_ram - start_ram
        assert ram_used < 10_000_000  # less than 10 MB of RAM should be used

    def test_time_to_train(self, df) -> None:
        X, y = df.iloc[:, 0:-1], df.iloc[:, -1].values

        start_time = time.time()

        _ = train(X, y)

        # test for training time
        end_time = time.time()
        time_taken = end_time - start_time
        assert time_taken < 10  # it should take less than 10 seconds to train

