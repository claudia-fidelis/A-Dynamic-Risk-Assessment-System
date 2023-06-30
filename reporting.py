"""
Author: Cláudia M. Fidelis
Date: June, 2023
This script generates plots related to your ML model's performance
"""

import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from diagnostics import model_predictions


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

output_model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path']) 



##############Function for reporting
def score_model():
    '''
    Calculate a confusion matrix using the test data and the deployed model
    Write the confusion matrix to the workspace

    input: None 
    output: None
    '''

    df_test = pd.read_csv(os.path.join(os.getcwd(), test_data_path, "testdata.csv"))

    y = df_test['exited']
    y_pred = model_predictions(df_test)

    cm = metrics.confusion_matrix(y, y_pred)

    plt.figure(figsize=(10,7))
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', cmap="Blues", ax=ax) #annot=True to annotate cells, ftm='g' to disable scientific notation
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['Not Churned', 'Churned']); ax.yaxis.set_ticklabels(['Not Churned', 'Churned']);

    plt.savefig(os.path.join(os.getcwd(), output_model_path, "confusionmatrix.png"))

if __name__ == '__main__':
    score_model()
