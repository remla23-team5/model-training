from pathlib import Path
from click.testing import CliRunner

from src.data.download_data import main as download_data
from src.data.make_dataset import main as make_dataset
from src.models.train_model import main as train_model


class TestInfrastructure():

    def test_pipeline(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            download_result = runner.invoke(download_data, ['1bCFMWa1lgymQtj6vukXTrtfF47TeKQLu', 'test_file.tsv'])

            assert download_result.exit_code == 0
            assert Path("test_file.tsv").exists()

            preprocess_result = runner.invoke(make_dataset, ['test_file.tsv', 'test_file_preprocessed.tsv'])

            assert preprocess_result.exit_code == 0
            assert Path("test_file_preprocessed.tsv").exists()

            train_result = runner.invoke(train_model, ['test_file_preprocessed.tsv', 'test_model.pkl', 'test_metrics.json'])

            assert train_result.exit_code == 0
            assert Path("test_metrics.json").exists()
            assert Path("test_model.pkl").exists()

