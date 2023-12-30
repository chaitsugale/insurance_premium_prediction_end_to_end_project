import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from india_insurance.entity import config_entity,artifact_entity

from india_insurance.exception import IndiaInsuranceException
from india_insurance.logger import logging
from india_insurance.utils import save_object,evaluate_models
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# @dataclass
# class ModelTrainerConfig:
#     train_model_file_path = os.path.join('artifacts','india_insurance_model.pkl')

class ModelTrainer:
    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
                 data_transformation_artifact:config_entity.DataTransformationArtifact):
        # self.model_trainer_config = ModelTrainerConfig()
        try:
            logging.info(f"{'>>'*20} Model trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise IndiaInsuranceException(e,sys)

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split Train and test data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest":RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "Linear Regression": LinearRegression(),
                "KNeighboursRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor" : DecisionTreeRegressor(),
                "XGBoost": XGBRegressor(),
            }

            params = {
                "Random Forest":{
                    'n_estimators':[8,16,32,64,128,256],
                    #'max_features':[1.0,'sqrt','log2']
                },
                "GradientBoostingRegressor": {
                    'n_estimators':[8,16,32,64,128,256],
                    #'max_features':[1.0,'sqrt','log2'],
                    'learning_rate':[.1,.01,.05,.001]
                },
                "AdaBoostRegressor": {
                    'n_estimators':[8,16,32,64,128,256],
                    #'max_features':[1.0,'sqrt','log2'],
                    'learning_rate':[.1,.01,.05,.001]
                },
                "Linear Regression": {},
                "KNeighboursRegressor": {},
                "DecisionTreeRegressor" : {'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2']},
                "XGBoost": {
                    'n_estimators':[8,16,32,64,128,256],
                    #'max_features':[1.0,'sqrt','log2'],
                    'learning_rate':[.1,.01,.05,.001]
                },
            }

            model_report:dict =evaluate_models(X_train=X_train,y_train=y_train,X_test =X_test,y_test=y_test,models = models
                                               ,params = params)

            #to get the bewst model score from dict

            best_model_score = max(sorted(model_report.values()))
            return best_model_score

            # to get the best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            return best_model_name

            best_model = models[best_model_name]
            return best_model

            if best_model_score < self.model_trainer_config.expected_score:
                raise IndiaInsuranceException(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {best_model_score}")
            # logging.info(f"Best Model found on both training and testing dataset")

            if best_model_score>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff: {best_score_model} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            utils.save_object(
                file_path = self.model_trainer_config.model_path,
                obj=best_model
            )

            # predicted = best_model.predict(X_test)
            # r2_square = r2_score(y_test,predicted)
            # return r2_square

            #Preparing artifact
            model_trainer_artifact = artifact_entity.ModelTraninerArtifact(model_path = model_trainer_config.model_path)
            return model_trainer_artifact

        except Exception as e:
            raise IndiaInsuranceException(e,sys)