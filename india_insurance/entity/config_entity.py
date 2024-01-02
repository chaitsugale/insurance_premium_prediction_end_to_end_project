import os,sys
from india_insurance.exception import IndiaInsuranceException
from india_insurance.logger import logging
from datetime import datetime

FILE = 'insurance.csv'
TRAIN_FILE = 'india_insurance_train.csv'
TEST_FILE = 'india_insurance_test.csv'
TRANSFORMER_OBJECT_FILE = 'india_insurance_transformer.pkl'
TARGET_ENCODER_OBJECT_FILE = 'india_insurance_target_encoder.pkl'
MODEL_FILE = 'india_insurance_best_model.pkl'

class TrainingPipelineConfig:
    def __init__(self):
        try:
            self.artifact_directory = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception as e:
            raise IndiaInsuranceException(e,sys)
        
class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try :
            self.database_name="india_insurance"
            self.collection_name="premium"
            self.data_ingestion_directory = os.path.join(training_pipeline_config.artifact_directory,"data_ingestion")
            self.feature_store_file_path = os.path.join(self.data_ingestion_directory,"feature_store",FILE)
            self.train_data_path = os.path.join(self.data_ingestion_directory,"dataset",TRAIN_FILE)
            self.test_data_path = os.path.join(self.data_ingestion_directory,"dataset",TEST_FILE)
            self.test_size = 0.25
        except Exception as e:
            raise IndiaInsuranceException(e,sys)
        
    def to_dict(self,)->dict:
        try:
            return self.__dict__
        except Exception as e:
            raise IndiaInsuranceException(e,sys)
        
class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_directory = os.path.join(training_pipeline_config.artifact_directory,"data_validation")
        self.report_file_path = os.path.join(self.data_validation_directory,"india_insurance_report.yaml")
        self.missing_threshold:float = 0.25
        self.base_file_path  = os.path.join("insurance_main_dataset.csv")

class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_directory = os.path.join(training_pipeline_config.artifact_directory,"data_transformation")
        self.transform_object_path = os.getcwd(self.data_transformation_directory,"transformer",TRANSFORMER_OBJECT_FILE)
        self.transformed_train_path = os.getcwd(self.data_transformation_directory,"transformed",TRAIN_FILE.replace("csv","npz"))
        self.transformed_test_path = os.getcwd(self.data_transformation_directory,"transformed",TEST_FILE.replace("csv","npz"))
        self.target_encoder_path = os.getcwd(self.data_transformation_directory,"target_encoder",TARGET_ENCODER_OBJECT_FILE)

class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_directory = os.path.join(training_pipeline_config.artifact_directory,"model_trainer")
        self.model_path = os.path.join(self.model_trainer_directory,"model",MODEL_FILE)
        self.expected_score = 0.75
        self.overfitting_threshold = 0.3


class ModelPusherConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_pusher_directory = os.path.join(training_pipeline_config.artifact_directory,"model_pusher")
        self.saved_model_directory = os.path.join("saved_models")
        self.pusher_model_directory = os.path.join(self.model_pusher_directory,"saved_models")
        self.pusher_model_path = os.path.join(self.pusher_model_directory,MODEL_FILE)
        self.pusher_transformer_path = os.path.join(self.pusher_model_directory,TRANSFORMER_OBJECT_FILE)

class ModelEvaluationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.change_threshold = 0.02


        
