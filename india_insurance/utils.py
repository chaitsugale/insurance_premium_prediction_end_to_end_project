import os
import sys
import numpy as np
import pandas as pd

import dill

from india_insurance.exception import IndiaInsuranceException
from india_insurance.logger import logging
from india_insurance.config import mongo_client
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def get_collection_as_dataframe(database_name:str,collection_name:str)->pd.DataFrame:
    try:
        logging.info(f"Reading the data from database: {database_name} and collections:{collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Found columns: {df.columns}")
        if "_id" in df.columns:
            logging.info(f"Dropping columns : _id")
            df = df.drop("_id",axis = 1)
        logging.info(f"Row and columns in df:{df.shape}")
        return df
    except Exception as e:
        raise IndiaInsuranceException(e,sys)

def convert_columns_float(df:pd.DataFrame,exclude_columns:list)->pd.DataFrame:
    try:
        for column in df.columns:
            if column not in exclude_columns:
                if df[column].dtypes != 'O':
                    df[column]=df[column].astype('float')
        return df
    except Exception as e:
        raise IndiaInsuranceException(e,sys)
    
        


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok = True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise IndiaInsuranceException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
            raise IndiaInsuranceException(e,sys)

def save_numpy_array_data(file_path:str,array:np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise IndiaInsuranceException(e,sys)
    
def load_numpy_array_data(file_path:str)->np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise IndiaInsuranceException(e, sys)
    
# def evaluate_models(X_train,y_train,X_test,y_test,models,params):
#     try:
#         report = {}
#         for i in range(len(list(models))):
#             model = list(models.values())[i]
#             para = params[list(models.keys())[i]]

#             gs = GridSearchCV(model,para,cv = 5)
#             gs.fit(X_train,y_train)

#             model.set_params(**gs.best_params_)
#             model.fit(X_train,y_train)

#             y_train_pred = model.predict(X_train)

#             y_test_pred = model.predict(X_test)

#             train_model_score = r2_score(y_train,y_train_pred)

#             test_model_score = r2_score(y_test,y_test_pred)

#             report[list(models.keys())[i]] = test_model_score
#         return report
    
#     except Exception as e:
#         raise IndiaInsuranceException(e,sys)

