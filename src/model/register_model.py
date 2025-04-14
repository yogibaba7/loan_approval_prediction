import pandas as pd 
import numpy as np 

import json
import logging
import os
import sys

# configure logging
logger = logging.getLogger('model_evaluations')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

#CONFIGURE EXPERIMENT
import mlflow
import dagshub

dagshub.init(repo_owner='yogibaba7', repo_name='loan_approval_prediction', mlflow=True)
# set tracking uri
mlflow.set_tracking_uri('https://dagshub.com/yogibaba7/loan_approval_prediction.mlflow/')

# load model info
def load_model_info(file_path:str)->dict:
    logger.debug(f"loading the model info file from {file_path}")
    try:
        with open(file_path,'r') as f:
            file = json.load(f)
        return file
    except Exception as e:
        logger.error(f"error occured --> {e}")

# register model
def register_model(model_name:str,model_info:dict):
    logger.debug('registering the model')
    try:
        # # create model uri
        # model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # # register the model
        # model_version = mlflow.register_model(model_uri,model_name)

        # assign a alias to register model
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name,stages=["None"])[0]
        client.set_registered_model_alias(model_name,'production',latest_version.version)
        # assign a tag for current model 
        client.set_registered_model_tag(model_name,'testing','passed')

    except Exception as e:
        logger.error(f"error occured -> {e}")

# main 
def main():
    # model info file path
    model_info_path = 'reports/model_info.json'
    # get the model info
    model_info = load_model_info(model_info_path)
    # set model name
    model_name = 'mymodel'
    # register the model
    register_model(model_name,model_info)

if __name__=="__main__":
    main()






