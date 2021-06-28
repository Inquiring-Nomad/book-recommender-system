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


def _already_ran(entry_point_name, parameters, git_commit, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = mlflow.tracking.MlflowClient()
    all_run_infos = reversed(client.list_run_infos(experiment_id))
    for run_info in all_run_infos:
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in parameters.items():
            run_value = full_run.data.params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue

        if run_info.to_proto().status != RunStatus.FINISHED:
            eprint(
                ("Run matched, but is not FINISHED, so skipping " "(run_id=%s, status=%s)")
                % (run_info.run_id, run_info.status)
            )
            continue

        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        if git_commit != previous_version:
            eprint(
                (
                    "Run matched, but has a different source version, so skipping "
                    "(found=%s, expected=%s)"
                )
                % (previous_version, git_commit)
            )
            continue
        return client.get_run(run_info.run_id)
    eprint("No matching run has been found.")
    return None


# TODO(aaron): This is not great because it doesn't account for:
# - changes in code
# - changes in dependant steps
def _get_or_run(entrypoint, parameters, git_commit, use_cache=True):
    existing_run = _already_ran(entrypoint, parameters, git_commit)
    if use_cache and existing_run:
        print("Found existing run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
        return existing_run
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


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
