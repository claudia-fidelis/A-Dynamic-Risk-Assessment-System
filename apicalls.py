import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"


with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'])
test_file = os.path.join(test_data_path,'testdata.csv')

model_path = os.path.join(config['output_model_path']) 


#Call each API endpoint and store the responses
response1 = requests.post(URL + '/prediction' + f'?data_path={test_file}').content
response2 = requests.get(URL + '/scoring').content
response3 = requests.get(URL + '/summarystats').content
response4 = requests.get(URL + '/diagnostics').content


#combine all API responses
responses = {'Predictions':response1.decode('utf-8'),
            'Scoring':response2.decode('utf-8'),
            'Statistics':response3.decode('utf-8'),
            'Diagnostics':response4.decode('utf-8')}
final_responses = json.dumps(responses, indent = 4)

#write the responses to your workspace
responses_path = os.path.join(model_path,'apireturns.txt')
with open(responses_path,'w') as f:
    f.write(json.dumps(final_responses))


