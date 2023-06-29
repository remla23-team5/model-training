model-training
==============================

Disclaimer: badges are updated each Friday, status may not reflect current state of the repository.
![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/LucaCras/7c841f9e8e8c5e2fbe202b7a9758d798/raw/7645f71b88cac167fd7327d0c9454a2357de5819/codecov.json) ![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/LucaCras/7c841f9e8e8c5e2fbe202b7a9758d798/raw/baa5c69fcd3cd65a0fe57113d58d3c9b759fc5f5/test-results.json)

Welcome to model-training, the place where the restaurant sentiment analysis model for the remla23 project is trained

There are 2 ways of running the project.

## 1. DVC

(Optional) To check the DVC pipeline / directed acyclical graph (DAG), use:

```shell
dvc dag
```

(Optional) To check the DAG of the dependencies of all the artifacts, use:

```shell
dvc dag --outs
```

To run the experiment, use:

```shell
dvc exp run
```

To check the experiment log, use:

```shell
dvc exp show
```
## 2. Make

(Optional) Create & Activate virtual environment.

To install requirements, use:
```make
make requirements
```

To run linter, use:
```make
make lint
```

In order to (only) preprocess the data, use:
```make
make data
```

In order to train the model (depends on previous step!), use:
```make
make train
```

In order to run the tests, use:
```make
make test
```

other commands can be found in Makefile.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

