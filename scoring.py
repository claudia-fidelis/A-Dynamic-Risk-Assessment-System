from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



################# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path']) 


#explanatory variables and target
features = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
target = 'exited'

################# Function for model scoring
def score_model(test_data_path=None):
    '''
    Take the trained model, load test data and calculate an F1 score for the model relative to the test data
    Write the result to the latestscore.txt file

    input: None
    output: Final file latestscore.txt
    '''

    #load trained model
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)

    #load test data
    if test_data_path is None:
        test_data_path = os.path.join(config['test_data_path']) 
        df_test = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
        df_test = df_test.drop(columns=['corporation'])
    else:
        df_test = pd.DataFrame()
        filenames = os.listdir(os.path.join(os.getcwd(), test_data_path))
        for each_filename in filenames:
            #append if the file has a extension .csv
            if each_filename.endswith(".csv"):
                df = pd.read_csv(f'{os.getcwd()}/{test_data_path}/{each_filename}')
                df_test=pd.concat([df_test, df])

    X_test = df_test[features]
    y_test = df_test[target]

    #score test data and calculate F1 score
    y_pred = model.predict(X_test)
    f1score = metrics.f1_score(y_pred, y_test)

    #save the f1 score
    with open(os.path.join(model_path, "latestscore.txt"), "w") as score_file:
        score_file.write(str(f1score) + "\n")
    
    return f1score

if __name__ == '__main__':
    f1 = score_model()
    print(f1)