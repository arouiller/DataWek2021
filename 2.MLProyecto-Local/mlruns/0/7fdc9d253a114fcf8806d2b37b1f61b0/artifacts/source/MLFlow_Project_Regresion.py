import os
import sys
import click
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


@click.command()
@click.option('--alpha', type=float, help='alpha parameter')
@click.option('--l1_ratio', type=float, help='l1 ratio parameter')
@click.option('--data_file', default='', type=str, help='Data file')


def run(alpha, l1_ratio, data_file):#, l1_ratio, data_file, experiment_name, run_name):
    #tratamiento especial de la cadena porque viene con ' inicial y final
    if data_file[0] == "'":
        data_file = data_file[1:]
    if data_file[-1] == "'":
        data_file = data_file[:-1]

    data = pd.read_csv(data_file, sep = ';')

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    #exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    if 'images' not in os.listdir():
        os.mkdir ('images')
        
    #with mlflow.start_run(experiment_id=exp_id):
        
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    mlflow.log_params({'alpha': alpha, 'l1_ratio':l1_ratio})

    mlflow.log_metrics({'rmse': rmse, 'mae':mae, 'r2':r2})
    
    mlflow.set_tags({'ProblemType':'Regresion', 'ModelType':'Elasticnet', 'ModelLibrary':'Scikit-Learn', 'developer':'arouiller'})
    
    mlflow.sklearn.log_model(lr, "PlainRegression_Model")
    
    train_x.plot(kind='box', subplots=True, layout=(3,4), figsize=(16,9), title='Box plot of each feature')
    plt.savefig('images/distribution_plot_all_features.png')
    
    mlflow.log_artifacts('images', 'images')
    
    mlflow.log_artifact(__file__, 'source')


if __name__ == "__main__":
    run()