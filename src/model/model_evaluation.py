import pandas as pd 
import numpy as np 
import os 
import sys
import logging
import joblib
import yaml
import json
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

# configure logging
logger = logging.getLogger('model_evaluations')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# load data
def load_data(path:str)->tuple[pd.DataFrame]:
    logger.debug(f"loading data from {path}")
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logger.error(f"error occured --> {e}")

# load params

def load_params()->dict:
    logger.debug(f"loading parameters")
    try:
        with open('params.yaml') as f:
            file = yaml.safe_load(f)
        parameter = file['model_building']
        return parameter
    except Exception as e:
        logger.error(f"error occured -> {e}")

# load model
def load_model():
    logger.debug('Loading the model')
    try:
        model = joblib.load('models/model.pkl')
        return model
    except Exception as e:
        logger.error(f"error occured -> {e}")

# evaluate_results
def evaluate_results(model,X_test,y_test):
    try:
        logger.debug('making prediction..')
        y_pred = model.predict(X_test)
        logger.debug('evaluating results')
        accuracy = accuracy_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)

        dict = {
            'accuracy_score':accuracy,
            'f1_score':f1,
            'recall_score':recall,
            'precision_score':precision
        }
        logger.debug('saving result in reports.metrics.json')
        with open('reports/metrics.json','w') as f:
            json.dump(dict,f)
        return dict
    except Exception as e:
        logger.error(f"error occured -> {e}")

# save model_info
def save_model_info(run_id,model_path,file_path):
    logger.debug(f"saving the model info in {file_path}")
    try:
        model_info = {'run_id':run_id,'model_path':model_path}
        with open(file_path,'w') as f:
            json.dump(model_info,f,indent=4)
    except Exception as e:
        logger.error(f"error occured -->{e}")

#CONFIGURE EXPERIMENT
import mlflow
import dagshub
from mlflow.models.signature import infer_signature
dagshub.init(repo_owner='yogibaba7', repo_name='loan_approval_prediction', mlflow=True)
# set tracking uri
mlflow.set_tracking_uri('https://dagshub.com/yogibaba7/loan_approval_prediction.mlflow/')
# create a experiment
mlflow.set_experiment('pipeline_exp')

# main
def main():
    with mlflow.start_run(description='This is the run of best model with best parameters') as run:

        data_path = 'data/processed/test_processed.csv'
        # load testing data  for model evaluation
        test_data = load_data(data_path)
        # make x_test and y_test
        X_test = test_data.drop(columns=['Loan_Status'])
        y_test = test_data['Loan_Status']
        # load model
        model = load_model()
        # evaluate results
        results = evaluate_results(model,X_test,y_test)
        # load parameters of model_building
        parameters = load_params()

        # create model signature
        model_signature = infer_signature(X_test,y_test)
        # log model
        mlflow.sklearn.log_model(model,'AdaBoostClassifier',signature=model_signature)
        # log model parameters
        mlflow.log_params(parameters)
        # log model result metrics
        mlflow.log_metrics(results)
        # log metrics file to mlflow
        mlflow.log_artifact('reports/metrics.json')
        # set some tags 
        mlflow.set_tag('author','Yogesh chouhan')
        # save the model info
        run_id = run.info.run_id
        model_path = 'AdaBoostClassifier'
        save_model_info(run_id=run_id,model_path=model_path,file_path='reports/model_info.json')
        # log the model_info to mlflow
        mlflow.log_artifact('reports/model_info.json')

        

if __name__=="__main__":
    main()






