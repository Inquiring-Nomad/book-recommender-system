# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import urllib.request
from zipfile import ZipFile
from io import BytesIO
import mlflow
import pandas as pd
import os


@click.command()
@click.argument('input_dir', type=click.Path())
@click.argument('output_dir', type=click.Path())
def main(input_dir, output_dir):
    """ Download dataset.
    """
    logger = logging.getLogger(__name__)
    logger.info('processing dataset')

    books_df = pd.read_csv(os.path.join(input_dir, 'BX-Books.csv'), usecols=['ISBN', 'Book-Title'], sep=';',
                           error_bad_lines=False, encoding='latin-1')
    ratings_df = pd.read_csv(os.path.join(input_dir, 'BX-Book-Ratings.csv'), sep=';', error_bad_lines=False,
                             encoding='latin-1')
    users_df = pd.read_csv(os.path.join(input_dir, 'BX-Users.csv'), sep=';', error_bad_lines=False, encoding='latin-1')
    ratings_df = ratings_df[ratings_df['Book-Rating'] > 0]
    rating_books = pd.merge(ratings_df, books_df, on="ISBN", how='left')
    rating_books = rating_books[~(rating_books.duplicated(subset=["User-ID", "Book-Title"], keep=False) & rating_books[
        'Book-Title'].notnull())].sort_values(by="User-ID", ascending=False)
    grp_bks = rating_books.groupby('User-ID').agg({"Book-Rating": "count"}).sort_values(by="Book-Rating",
                                                                                        ascending=False)
    grp_bks_df = grp_bks[grp_bks['Book-Rating'] > 100]
    grp_rtg = rating_books.groupby('ISBN').agg({"Book-Rating": "count"}).sort_values(by="Book-Rating", ascending=False)
    grp_rtg_df = grp_rtg[grp_rtg['Book-Rating'] > 50]
    rating_books = rating_books[rating_books['User-ID'].isin(grp_bks_df.index)]
    rating_books = rating_books[rating_books['ISBN'].isin(grp_rtg_df.index)]
    logger.info(f'New dataset shape: {rating_books.shape}')
    processed_path = os.path.join(output_dir,'rating_books.csv')
    rating_books.to_csv(processed_path)
    mlflow.log_artifact(processed_path)




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
