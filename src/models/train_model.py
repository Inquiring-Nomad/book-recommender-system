# -*- coding: utf-8 -*-
import click
import logging

import mlflow.sklearn
from dotenv import find_dotenv, load_dotenv
from surprise import Dataset, Reader, KNNWithMeans, accuracy, SVD, KNNBasic
from surprise.model_selection import cross_validate, train_test_split
import pandas as pd
import random
import os
import numpy as np


@click.command()
@click.argument('input_dir', type=click.Path())
@click.argument('output_dir', type=click.Path())
@click.argument('seed', type=click.FLOAT)
@click.argument('split', type=click.FLOAT)
@click.argument('similarity', type=click.STRING)
@click.argument('user_based', type=click.BOOL)
def main(input_dir, output_dir, seed, split, similarity, user_based):
    """ Train the model.
    """
    np.random.seed(int(seed))
    logger = logging.getLogger(__name__)
    logger.info('processing dataset')

    rating_books = pd.read_csv(os.path.join(input_dir, 'rating_books.csv'),
                               error_bad_lines=False, encoding='latin-1')
    reader = Reader(rating_scale=(1, 10))

    # Loads Pandas dataframe
    data = Dataset.load_from_df(rating_books[["User-ID", "ISBN", "Book-Rating"]], reader)
    raw_ratings = data.raw_ratings
    random.shuffle(raw_ratings)
    threshold = int(split * len(raw_ratings))
    train_raw_ratings = raw_ratings[:threshold]
    test_raw_ratings = raw_ratings[threshold:]

    data.raw_ratings = train_raw_ratings  # data is now train
    # mlflow.sklearn.autolog()
    sim_options = {
        "name": similarity,
        "user_based": user_based,  # Compute  similarities between items
    }
    testset = data.construct_testset(test_raw_ratings)

    with mlflow.start_run(run_name="knn-mean") as run:
        runId = run.info.run_id
        mlflow.set_tag('mlflow.runName', "knn-mean")
        knnmeans = KNNWithMeans(sim_options=sim_options)
        logger.info('cross-validate knnmeans')
        knnmeansresults = cross_validate(knnmeans, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
        mlflow.log_metric('crossval_mae', knnmeansresults['test_mae'].mean())
        mlflow.log_metric('crossval_rmse', knnmeansresults['test_rmse'].mean())
        predictions = knnmeans.test(testset)
        logger.info('evaluating knnmeans')
        test_rmse = accuracy.rmse(predictions)
        test_mae = accuracy.mae(predictions)
        mlflow.log_metric('test_mae', test_mae)
        mlflow.log_metric('test_rmse', test_rmse)
        mlflow.sklearn.log_model(
            sk_model=knnmeans,
            artifact_path='knnmeans',
            registered_model_name="knnmeans",

        )
        with mlflow.start_run(nested=True, run_name="knn-basic") as run:
            knnbasic = KNNBasic(sim_options=sim_options)
            knnbasicresults = cross_validate(knnbasic, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
            mlflow.log_metric('crossval_mae', knnbasicresults['test_mae'].mean())
            mlflow.log_metric('crossval_rmse', knnbasicresults['test_rmse'].mean())
            predictions = knnbasic.test(testset)
            logger.info('evaluating knnbasic')
            test_rmse = accuracy.rmse(predictions)
            test_mae = accuracy.mae(predictions)
            mlflow.log_metric('test_mae', test_mae)
            mlflow.log_metric('test_rmse', test_rmse)
            mlflow.sklearn.log_model(
                sk_model=knnbasic,
                artifact_path='knnbasic',
                registered_model_name="knnbasic",

            )
        with mlflow.start_run(nested=True, run_name="svd") as run:
            svd = SVD()
            svdresults = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
            mlflow.log_metric('crossval_mae', svdresults['test_mae'].mean())
            mlflow.log_metric('crossval_rmse', svdresults['test_rmse'].mean())
            predictions = svd.test(testset)
            logger.info('evaluating svd')
            test_rmse = accuracy.rmse(predictions)
            test_mae = accuracy.mae(predictions)
            mlflow.log_metric('test_mae', test_mae)
            mlflow.log_metric('test_rmse', test_rmse)
            mlflow.sklearn.log_model(
                sk_model=svd,
                artifact_path='svd',
                registered_model_name="svd",

            )

    # with mlflow.start_run(nested=True,run_name="knn-basic",run_id=runId) as run:
    #     knnbasic = KNNBasic(sim_options=sim_options)
    #     knnbasicresults = cross_validate(knnbasic, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
    #     mlflow.log_metric('crossval_mae', knnbasicresults['test_mae'].mean())
    #     mlflow.log_metric('crossval_rmse', knnbasicresults['test_rmse'].mean())
    #     predictions = knnbasic.test(testset)
    #     logger.info('evaluating knnbasic')
    #     test_rmse = accuracy.rmse(predictions)
    #     test_mae = accuracy.mae(predictions)
    #     mlflow.log_metric('test_mae', test_mae)
    #     mlflow.log_metric('test_rmse', test_rmse)
    #     mlflow.sklearn.log_model(
    #         sk_model=knnbasic,
    #         artifact_path='knnbasic',
    #         registered_model_name="knnbasic",
    #
    #     )
    # with mlflow.start_run(nested=True,run_name="svd",run_id=runId) as run:
    #     svd = SVD()
    #     svdresults = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
    #     mlflow.log_metric('crossval_mae', svdresults['test_mae'].mean())
    #     mlflow.log_metric('crossval_rmse', svdresults['test_rmse'].mean())
    #     predictions = svd.test(testset)
    #     logger.info('evaluating svd')
    #     test_rmse = accuracy.rmse(predictions)
    #     test_mae = accuracy.mae(predictions)
    #     mlflow.log_metric('test_mae', test_mae)
    #     mlflow.log_metric('test_rmse', test_rmse)
    #     mlflow.sklearn.log_model(
    #         sk_model=svd,
    #         artifact_path='svd',
    #         registered_model_name="svd",
    #
    #     )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
