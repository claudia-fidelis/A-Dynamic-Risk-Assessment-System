import training
import scoring
import deployment
import diagnostics
import reporting
import json
import os
import subprocess


with open("config.json", "r") as f:
    config = json.load(f)


input_folder_path = config["input_folder_path"]
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])



##################Check and read new data
#first, read ingestedfiles.txt

ingested_file_path = os.path.join(prod_deployment_path, "ingestedfiles.txt")
ingested_files = []
with open(ingested_file_path, "r") as f:
    for line in f:
        ingested_files.append(line.rstrip())
        
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt

new_files = []
for filename in os.listdir(input_folder_path):
    if filename not in ingested_files:
        new_files.append(filename.rstrip())


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here

if not new_files:
    print('We did not found new data, exiting')
    exit()

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data

with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'rb') as f:
    latest_score = float(f.read())

print(f"Last F1 score: {latest_score}")

f1_new = scoring.score_model(input_folder_path)
print('New F1 score: ', f1_new)

##################Deciding whether to proceed, part 2
#If f1_new is lower then latest_score, model drift has occurred
#if you found model drift, you should proceed. otherwise, do end the process here

if f1_new >= latest_score:
    exit()

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

training.train_model(input_folder_path)
deployment.store_model_into_pickle()

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model

reporting.score_model()
subprocess.call("python apicalls.py", shell=True)



