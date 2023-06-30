"""
Author: Cláudia M. Fidelis
Date: June, 2023
This script does model and data diagnostics
"""

import pandas as pd
import numpy as np
import timeit
import os
import sys
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

#explanatory variables
features = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']



##################Function to get model predictions
def model_predictions(df=None):
    '''
    Read the deployed model and a test dataset.
    Calculate predictions.

    input: A pandas DataFrame 
    output: A list containing all predictions
    '''

    #load trained model
    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)

    #if no data frame was provided
    if df is None:
        test_path = f"{os.getcwd()}/{test_data_path}/testdata.csv"
        df = pd.read_csv(test_path)

    #score test data
    y_pred = model.predict(df[features])

    return y_pred



##################Function to get summary statistics
def dataframe_summary():
    '''
    Calculate summary statistics (means, medians, and standard deviations)

    input: None
    output: A list containing all summary statistics
    '''

    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))

    numeric_features = df.select_dtypes(include=np.number)
    if 'corporation' in numeric_features: 
        numeric_features.remove('corporation')

    #calculate summary statistics here
    # statistics = {}
    statistics = []
    for col in numeric_features:   
    #     mean = round(df[col].mean(),2)
    #     median = round(df[col].median(),2)
    #     std = round(df[col].std(),2)

    #     statistics[col] = {'MEAN': mean, 'MEDIAN': median, 'STD': std}

    # statistics = pd.DataFrame.from_dict(statistics).transpose()
    # statistics = statistics.rename_axis('FEATURES').reset_index()

        statistics.append([col, "mean", df[col].mean()])
        statistics.append([col, "median", df[col].median()])
        statistics.append([col, "standard deviation", df[col].std()])

    return statistics



##################Function to get percentage of missing data
def missing_data():
    '''
    Calculate percentage of missing

    input: None
    output: A list containing percentage of missing data
    '''

    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))

    nulls = []

    for col in features:
        perc_null = df[col].isnull().sum()/len(df)
        nulls.append([col, str(perc_null) + "%"])

    return str(nulls)



##################Function to get timings
def execution_time():
    '''
    Calculate timing of training.py and ingestion.py

    input: None
    output: A list with timing value in seconds of data ingestion and model training
    '''
    
    result = []
    for procedure in ["training.py" , "ingestion.py"]:
        starttime = timeit.default_timer()
        os.system('python3 %s' % procedure)
        timing=timeit.default_timer() - starttime
        result.append([procedure, timing])
 
    return str(result)



##################Function to check dependencies
def outdated_packages_list():
    '''
    Checks each package status if it is outdated or not
    '''

    outdated_packages = subprocess.check_output(['pip', 'list', '--outdated']).decode(sys.stdout.encoding)

    return str(outdated_packages)



if __name__ == '__main__':
    model_predictions()
    print(dataframe_summary())
    print(missing_data())
    print(execution_time())
    print(outdated_packages_list())






    
