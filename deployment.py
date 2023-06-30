from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path']) 


####################function for deployment
def store_model_into_pickle():
    '''
    Copy the latest pickle file, 
    the latestscore.txt value and 
    the ingestfiles.txt file into the deployment directory

    '''
        
    #create an output directory if it doesn't exist
    if not os.path.exists(os.path.join(os.getcwd(), prod_deployment_path)):    
        os.makedirs(os.path.join(os.getcwd(), prod_deployment_path))   
        
    shutil.copy(
        f"{os.getcwd()}/{model_path}/trainedmodel.pkl",
        f"{os.getcwd()}/{prod_deployment_path}/"
    )
    
    shutil.copy(
        f"{os.getcwd()}/{model_path}/latestscore.txt",
        f"{os.getcwd()}/{prod_deployment_path}/"
    )
    
    shutil.copy(
        f"{os.getcwd()}/{dataset_csv_path}/ingestedfiles.txt",
        f"{os.getcwd()}/{prod_deployment_path}/"
    )
        
if __name__ == '__main__':
    store_model_into_pickle()
