import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from india_insurance.logger import logging
from india_insurance.exception import IndiaInsuranceException
from india_insurance.utils import save_object
from india_insurance.config import TARGET_COLUMN


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','india_insurance_preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    @classmethod
    def get_data_transformer_object(self)->Pipeline:
        '''
        This function is responsible for 
        '''
        
        try:
            simple_imputer = SimpleImputer(strategy = 'constant',fill_value =0)
            robust_scaler = RobustScaler()
            pipeline = Pipeline(steps=[
                ('imputer',simple_imputer),
                ('robust_scaler',robust_scaler)
                ])
            return pipeline
        except Exception as e:
            raise IndiaInsuranceException(e,sys)
        
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            #preprocessing_obj = self.get_data_transformer_object()

            #target_column_name = "expenses"
            #numerical_columns = ["age","bmi","children","expenses"]

            input_feature_train_df = train_df.drop(TARGET_COLUMN,axis = 1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns = [TARGET_COLUMN],axis = 1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe." 
            )

            label = LabelEncoder()

            #transformation of target column
            target_feature_train_arr  = target_feature_train_df.squeeze()
            target_feature_test_arr  = target_feature_test_df.squeeze()

            #transformation on categorical columns

            for col in input_feature_train_df.columns:
                if input_feature_test_df[col].dtype =='O':
                    input_feature_train_df[col] = label.fit_transform(input_feature_train_df[col])
                    input_feature_test_df[col] = label.fit_transform(input_feature_test_df[col])
                else:
                    input_feature_train_df[col] = input_feature_train_df[col]
                    input_feature_test_df[col] = input_feature_test_df[col]

            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)


            '''
            difference between fit_transform and transform -->

            fit_transform: This method is used to both fit the transformation model to the data and apply the transformation in a single step.

            transform : This method is used to apply a previously learned transformation model to new or unseen data.
            '''
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

            '''
            use of np.c_ --> The np.c_ object is especially useful when you want to concatenate multiple arrays along the second axis
            '''

            train_ar = np.c_[
                input_feature_train_arr ,np.array(target_feature_train_df)

            ]

            test_ar = np.c_[
                input_feature_test_arr ,np.array(target_feature_test_df)
                
            ]

            logging.info(f"saved preprocessing obj")

            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path,
                     obj = transformation_pipeline)

            return(
                train_ar,test_ar,self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise IndiaInsuranceException(e,sys)