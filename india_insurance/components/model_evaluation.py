from india_insurance.predictor import ModelResolver
from india_insurance.entity import config_entity,artifact_entity
from india_insurance.exception import IndiaInsuranceException
from india_insurance.logger import logging
from india_insurance.utils import load_object
from sklearn.metrics import r2_score
import pandas as pd
import sys,os
from india_insurance.config import TARGET_COLUMN

class ModelEvaluation:
    def __init__(self,
                 model_eval_config:config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact:artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact:artifact_entity.ModelTraninerArtifact):
        try:
            logging.info(f"{'>>'*20}  Model Evaluation {'<<'*20}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise IndiaInsuranceException(e,sys)
        
    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            pass
        except:
            pass