Book Recommender
==============================

A book recommendation system , trained with the collaborative-filtering technique.

Three algorithms have been used,
- KNNMeans, KNNBasic
- matrix-factorisation with  SVD

##### Tools and Packages:

- [Cookiecutter](https://drivendata.github.io/cookiecutter-data-science/) is used to generate the project folder structure
- The experiments and the artifacts are tracked with [MLFlow](https://www.mlflow.org/docs/latest/index.html)
- For the training and evaluation of the models I am using the [Surprise](https://surprise.readthedocs.io/en/stable/index.html) package.
- [Click](https://click.palletsprojects.com/en/8.0.x/) is used to help with the command-line interface.

##### Dataset:

[Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)

**Starting the MLFlow tracking server**

mlflow server /\
--backend-store-uri sqlite:///mlflow.db /\
--default-artifact-root artifacts /\
--host 127.0.0.1 -p 1234

**Commands:**

The basic command to run the experiments is :

`mlflow run  . --no-conda --experiment-name 'Book recommendations'`

Several options can be added as parameters, for example:

`mlflow run  . --no-conda --experiment-name 'Book recommendations' -P get_data=True -P process_data=True train-model=True`

Will re-download and reprocess the dataset and will re-train the models, adding them as new model versions in the MLFlow model registry.

By default the data directories:

- `data/external`
- `data/processed`

are not version controlled because [data is treated as immutable](https://drivendata.github.io/cookiecutter-data-science/#data-is-immutable) and they are generated on the fly by the experiments.

For a full list of options , please check the **MLproject** file










