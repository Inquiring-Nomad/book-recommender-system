"""
Downloads the MovieLens dataset, ETLs it into Parquet, trains an
ALS model, and uses the ALS model to train a Keras neural network.

See README.rst for more details.
"""

import os
import click
import mlflow
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint
from mlflow.tracking.fluent import _get_experiment_id
import logging
import pathlib


@click.command()
@click.option('--get-data', default=False, type=click.BOOL, show_default=True)
@click.option('--process-data', default=False, type=click.BOOL, show_default=True)
@click.option('--train-model', default=True, type=click.BOOL, show_default=True)
@click.option("-seed", default=42.0, type=click.FLOAT, help="Random seed", show_default=True)
@click.option("-split", default=.8, type=click.FLOAT, help="Train/test split", show_default=True)
@click.option("-similarity", default='cosine', type=click.STRING, help="The similarity type to be used",
              show_default=True)
@click.option("--user-based", default=False, type=click.BOOL, help="User based or item based", show_default=True)
def main(get_data, process_data, train_model, seed, split, similarity, user_based):
    pathlib.Path('data/external').mkdir(parents=True, exist_ok=True)
    pathlib.Path('data/processed').mkdir(parents=True, exist_ok=True)
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run() as active_run:
        if get_data:
            logger = logging.getLogger(__name__)
            logger.info('Download dataset set to true')
            download_data = mlflow.run(".", 'download_data', use_conda=False)
        if process_data:
            logger = logging.getLogger(__name__)
            logger.info('Process dataset set to true')
            process = mlflow.run(".", 'process_data', use_conda=False, )
        if train_model:
            logger = logging.getLogger(__name__)
            logger.info('Train model  set to true')
            train = mlflow.run(".", 'train', use_conda=False,
                               parameters={'seed': seed, 'similarity': similarity, 'user_based': user_based,
                                           'split': split})


if __name__ == "__main__":
    main()
