import pandas as pd 
import numpy as np
import logging
import os
import sys
import joblib

# sklearn libries
from sklearn.preprocessing import OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler

# configure logging
logger = logging.getLogger('data_preprocessing_log')
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

# save encoding
def save_encoding(data:pd.DataFrame)->None:
    logger.debug('training and saving the model in models/encoder.pkl')
    try:
        # train encoder
        oe = OrdinalEncoder()
        oe.fit(data)
        # save encoder
        joblib.dump(oe,'models/encoder.pkl')
    except Exception as e:
        logger.error(f"error occured -->{e}")

# apply encoding
def apply_encoding(data:pd.DataFrame)->pd.DataFrame:
    logger.debug('applying encoder on data')
    try:
        # load encoder
        model = joblib.load('models/encoder.pkl')
        data = model.transform(data)
        return data
    except Exception as e:
        logger.error(f"error occured --> {e}")

# save iterative imputer
def save_imputer(data:pd.DataFrame)->None:
    logger.debug('training and saving the model in models/imputer.pkl')
    try:
        # train imputer
        ii = IterativeImputer()
        ii.fit(data)
        # save imputer
        joblib.dump(ii,'models/imputer.pkl')
    except Exception as e:
        logger.error(f"error occured -->{e}")

# apply imputer
def apply_imputer(data:pd.DataFrame)->np.array:
    # load imputer
    logger.debug('applying imputer on data')
    try:
        model = joblib.load('models/imputer.pkl')
        data = model.transform(data)
        return data
    except Exception as e:
        logger.error(f"error occured -> {e}")

# save scaler 
def save_scaler(data:pd.DataFrame)->None:
    logger.debug('training and saving the scaler in models/scaler.pkl')
    try:
        # train robustscaler
        rs = RobustScaler()
        rs.fit(data)
        # save scaler
        joblib.dump(rs,'models/scaler.pkl')
    except Exception as e:
        logger.error(f"error occured ->{e}")

# apply scaler
def apply_scaler(data:pd.DataFrame)->pd.DataFrame:
    logger.debug('applying scaler on data')
    try:
        # load scaler
        model = joblib.load('models/scaler.pkl')
        # transform data
        data = model.transform(data)
        return data
    except Exception as e:
        logger.error(f"error occured -> {e}")

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


def main():
    # path of train and test data
    train_path = 'data/raw/train.csv'
    test_path = 'data/raw/test.csv'

    # load train data
    train_data = load_data(train_path)
    test_data = load_data(test_path)

    # categorical cols and numerical cols in our data
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

    # train encoder
    save_encoding(train_data[cat_cols])

    # apply encoder on both train and test set
    train_data[cat_cols] = apply_encoding(train_data[cat_cols])
    test_data[cat_cols] = apply_encoding(test_data[cat_cols])

    # train imputer
    save_imputer(train_data)

    # apply imputer on both train and test set . it will return array so we need to again convert them in dataframe for future uses.
    cols = train_data.columns
    train_data = pd.DataFrame(apply_imputer(train_data),columns=cols)
    test_data = pd.DataFrame(apply_imputer(test_data),columns=cols)

    # train scaler
    save_scaler(train_data[num_cols])

    # apply scaler
    train_data[num_cols] = apply_scaler(train_data[num_cols])
    test_data[num_cols] = apply_scaler(test_data[num_cols])

    # save path
    save_path = os.path.join('data','processed')
    # save data
    save_data(train_data,'train_processed.csv',save_path)
    save_data(test_data,'test_processed.csv',save_path)

if __name__=="__main__":
    main()







