import sys
import click
from numpy import str0
import mlflow
import pandas as pd

@click.command()
@click.option('--logged_model', default='', type=str, help='uri del modelo')
@click.option('--data_file', default='', type=str, help='dataset')


def run(logged_model, data_file):
    if logged_model[0] == "'":
        logged_model = logged_model[1:]
    if logged_model[-1] == "'":
        logged_model = logged_model[:-1]

    if data_file[0] == "'":
        data_file = data_file[1:]
    if data_file[-1] == "'":
        data_file = data_file[:-1]

    data = pd.read_csv(data_file, sep = ';')
    x = data.drop(["quality"], axis=1)
    y = data[["quality"]]

    loaded_model = mlflow.pyfunc.load_model(logged_model)
    y = loaded_model.predict(pd.DataFrame(x))

    print(y)


if __name__ == "__main__":
    run()