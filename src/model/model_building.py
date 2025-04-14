import pandas as pd 
import numpy as np 
import os 
import sys
import logging
import yaml
import joblib

from sklearn.ensemble import AdaBoostClassifier

# configure logging
logger = logging.getLogger('model_building')
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

# load params.yaml
def load_params(parameter:str)->float:
    logger.debug(f"loading {parameter}")
    try:
        with open('params.yaml') as f:
            file = yaml.safe_load(f)
        parameter = file['model_building'][parameter]
        return parameter
    except Exception as e:
        logger.error(f"error occured -> {e}")

# train model
def train_model(X_train:pd.DataFrame,y_train):
    logger.debug('training the model')
    try:
        # train model
        model = AdaBoostClassifier(n_estimators=load_params('n_estimators'),learning_rate=load_params('learning_rate'))
        model.fit(X_train,y_train)
        # save model
        joblib.dump(model,'models/model.pkl')
    except Exception as e:
        logger.error(f"error occured -> {e}")

# main
def main():
    train_path = 'data/processed/train_processed.csv'
    # laod data
    train_data = load_data(train_path)
    # make x_train and y_train
    X_train = train_data.drop(columns=['Loan_Status'])
    y_train = train_data['Loan_Status']

    # train model
    train_model(X_train,y_train)

if __name__=="__main__":
    main()

