""" 
In this script we read data files into Python and write them to an output file that will be your master dataset. We also save a record of the files we've read.
"""   

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


############# Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


############# Function for data ingestion
def merge_multiple_dataframe():
    """ 
    Read data files into Python and 
    write them to an output file that will be a master dataset,
    then we save a record of the files.

    input: None
    output: Final dataset and list of ingested files saved to disk
    """    
    directories = [input_folder_path]

    ingestedfiles = []
    finaldata = pd.DataFrame()
    
    for directory in directories:

        #list with the datasets
        filenames = os.listdir(os.path.join(os.getcwd(), directory))

        for each_filename in filenames:

            #append if the file has a extension .csv
            if each_filename.endswith(".csv"):
                df = pd.read_csv(f'{os.getcwd()}/{directory}/{each_filename}')
                ingestedfiles.append(each_filename)
                finaldata=pd.concat([finaldata, df])    

    #remove duplicates
    finaldata.drop_duplicates(inplace=True)

    #create an output directory if it doesn't exist
    if not os.path.exists(os.path.join(os.getcwd(), output_folder_path)):    
        os.makedirs(os.path.join(os.getcwd(), output_folder_path))

    #save the output file "finaldata.csv" in the output_folder_path
    finaldata.to_csv(os.path.join(output_folder_path,'finaldata.csv'), index=False)

    #save ingested files with timestamp
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), "w") as f:
        f.write(str(ingestedfiles)) 

if __name__ == '__main__':
    merge_multiple_dataframe()
