from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 


model_path = os.path.join(config['output_model_path']) 


#################Function for training the model
def train_model(dataset_csv_path=None):
    '''
    Read in finaldata.csv using the pandas module. 
    Use the scikit-learn module to train a logistic regression model for churn classification.
    Write the trained model to your workspace, in a file called trainedmodel.pkl

    input: None
    output: Final file trainedmodel.pkl
    '''
    
    #features to used in the training
    features = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
    target = 'exited'

    # import dataset
    if dataset_csv_path is None:
        dataset_csv_path = os.path.join(config['output_folder_path']) 
        filepath = os.path.join(dataset_csv_path, "finaldata.csv")
        df = pd.read_csv(filepath)

    else:
        df = pd.DataFrame()
        filenames = os.listdir(os.path.join(os.getcwd(), dataset_csv_path))
        for each_filename in filenames:
            #append if the file has a extension .csv
            if each_filename.endswith(".csv"):
                data = pd.read_csv(f'{os.getcwd()}/{dataset_csv_path}/{each_filename}')
                df = pd.concat([df, data])

    X = df[features]
    y = df[target]

    #use this logistic regression for training
    lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('model', lr)
    ])
    
    #fit the logistic regression to your data
    model = pipeline.fit(X, y)

    #create an output directory if it doesn't exist
    if not os.path.exists(os.path.join(os.getcwd(), model_path)):    
        os.makedirs(os.path.join(os.getcwd(), model_path))

    #write the trained model to your workspace in a file called trainedmodel.pkl
    savepath = os.path.join(model_path,'trainedmodel.pkl')
    with open(savepath, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    train_model()