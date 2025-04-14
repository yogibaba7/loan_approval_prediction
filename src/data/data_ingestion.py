import pandas as pd 
import numpy as np 
import logging
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import yaml
from sklearn.model_selection import train_test_split


# configure logging
logger = logging.getLogger('Data_ingestion_log')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# load params
def load_params(parameter):
    logger.debug('loading the parameters')
    try:
        with open('params.yaml','r') as f:
            file = yaml.safe_load(f)
        parameter = file['data_ingestion'][parameter]
        return parameter
    except Exception as e:
        logger.error(f"error occured --> {e}")
            


# load dataset
def load_data(url:str)->pd.DataFrame:
    logger.debug(f"loading data from {url}")
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        logger.error(f"error occured --> {e}")

# process_data
def process_data(data:pd.DataFrame)->pd.DataFrame:
    logger.debug(f"processing the data")
    try:
        data = data.drop(columns=['Loan_ID'])
        data["Loan_Status"] = data['Loan_Status'].replace({'Y':1,'N':0})
        return data
    except Exception as e:
        logger.error(f"error occured --> {e}")


# train test split
def create_trainset_testset(data:pd.DataFrame,test_size:float)->tuple[pd.DataFrame,pd.DataFrame]:
    logger.debug('creating training and testing set')
    try:
        train_data,test_data = train_test_split(data,test_size=test_size)
        return train_data,test_data
    except Exception as e:
        logger.error(f"error occured --> {e}")

# save data 
def save_data(data:pd.DataFrame,data_name:str,path:str):
    logger.debug(f"saving data {data_name} in directory {path}")
    try:
        # create a directory
        os.makedirs(path,exist_ok=True)
        # save data in directory
        data.to_csv(os.path.join(path,data_name),index=False)
    except Exception as e:
        logger.error(f"error occured --> {e}")


# main 
def main():
    url = 'https://raw.githubusercontent.com/rohitmande-inttrvu/finance_loan_approval/refs/heads/main/Finance.csv'
    df = load_data(url)
    df = process_data(df)

    # load testsize parameter
    test_size = load_params('test_size')
    train_data , test_data = create_trainset_testset(df,test_size)

    # save path
    save_path = os.path.join('data','raw')
    # save train data
    save_data(train_data,'train.csv',save_path)
    # save test data
    save_data(test_data,'test.csv',save_path)

if __name__=="__main__":
    main()