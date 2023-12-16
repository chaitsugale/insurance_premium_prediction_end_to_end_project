import sys
import pandas as pd
from india_insurance.exception import IndiaInsuranceException
from india_insurance.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts\india_insurance_model.pkl'
            preprocessor_path = 'artifacts\india_insurance_preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise IndiaInsuranceException(e,sys)

class CustomData:
    def __init__(
            self,
            age:int,
            sex:str,
            bmi:int,
            children:int,
            smoker:str,
            region:str,
            expenses:int
    ):
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region
        self.expenses = expenses

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
               "age " : [self.age], 
                "sex ": [self.sex],
                "bmi" : [self.bmi],
                "children":[self.children],
                "smoker" : [self.smoker],
                "region" :[self.region],
                "expenses" : [self.expenses],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise IndiaInsuranceException(e,sys)