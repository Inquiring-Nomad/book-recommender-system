# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import urllib.request
from zipfile import ZipFile
from io import BytesIO
import pathlib


@click.command()
@click.argument('url', type=click.STRING)
@click.argument('output_filepath', type=click.Path())
def main(url, output_filepath):
    """ Download dataset.
    """
    logger = logging.getLogger(__name__)
    logger.info('downloading dataset')

    with urllib.request.urlopen(url) as response:
        with ZipFile(BytesIO(response.read())) as zfile:
            zfile.extractall(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
