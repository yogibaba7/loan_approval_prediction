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
    logger.debug(f"loading {parameter}")
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

# main
def main():
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

if __name__=="__main__":
    main()






