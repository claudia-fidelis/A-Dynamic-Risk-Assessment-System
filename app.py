"""
Author: Cláudia M. Fidelis
Date: June, 2023
This script helps you to easily access ML diagnostics and results
"""

from flask import Flask, session, jsonify, request, Response
import pandas as pd
import numpy as np
import pickle

import json
import os

import diagnostics, scoring, ingestion



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    data_path = request.form.get('dataset_path')
    y_pred = diagnostics.model_predictions(data_path)

    return str(y_pred)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model and return F1 score number)
    score = scoring.score_model()
    return str(score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    statistics = diagnostics.dataframe_summary()
    return str(statistics)


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostic():        
    #check dependency, timing and percent NA values
    missing_data = diagnostics.missing_data()
    timing = diagnostics.execution_time()
    dependency_check = diagnostics.outdated_packages_list()

    # return str(["Execution_time:" + timing + "\nMissing_data;" + missing_data + "\nOutdated_packages:" + dependency_check])
    return Response(["Execution_time: ", str(timing), '\n\n', "Missing_data: ", str(missing_data), '\n\n', str(dependency_check)], content_type='text/plain', status=200)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
