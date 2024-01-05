import os
import sys
import numpy as np
from india_insurance.exception import IndiaInsuranceException
from india_insurance.logger import logging
import pandas as pd
from india_insurance import utils

from sklearn.model_selection import train_test_split
from dataclasses import dataclass #defining variable
from india_insurance.entity import artifact_entity,config_entity

from india_insurance.components.model_trainer import ModelTrainer

from india_insurance.components.data_transformation import DataTransformation

#any data input is required it will provided by data ingestion config class
# @dataclass
# class DataIngestionConfig:
#     train_data_path : str = os.path.join('artifacts',"train.csv")
#     test_data_path : str = os.path.join('artifacts',"test.csv")
#     raw_data_path : str = os.path.join('artifacts',"raw.csv") 

#other function inside the class then we can give init
class DataIngestion:
    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig):
        #3 files will get saved in this class variable
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise IndiaInsuranceException(e,sys)
    
    #reading the dataset, read in db then create mongodb client 
    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or components')
        try:
            logging.info(f"Exporting collection data as pandas dataframe")
            # df = pd.read_csv('notebook\insurance-premium-prediction\insurance.csv')

            df:pd.DataFrame = utils.get_collection_as_dataframe(
                database_name = self.data_ingestion_config.database_name,
                collection_name = self.data_ingestion_config.collection_name
            )
            logging.info("Save data in feature store")

            df.replace(to_replace="na",value = np.NAN,inplace=True)

            logging.info('Create feature store folder if not available')

            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir,exist_ok=True)
            logging.info("Save df to feature store folder")

            #Save df to feature store folder
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path,index=False,header=True)


            logging.info("Train test Split initiated")
            train_set,test_set = train_test_split(df,test_size = self.data_ingestion_config.test_size,random_state=42)

            dataset_dir = os.path.dirname(self.data_ingestion_config.train_data_path)
            os.makedirs(dataset_dir,exist_ok=True)

            train_set.to_csv(path_or_buf=self.data_ingestion_config.train_data_path,index = False,header = True)
            test_set.to_csv(path_or_buf=self.data_ingestion_config.test_data_path,index = False,header = True)

            logging.info('Ingestion of data is completed')

            # return(
            #     self.ingestion_config.train_data_path,
            #     self.ingestion_config.test_data_path
            # )

            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_data_path = self.data_ingestion_config.train_data_path,
                test_data_path = self.data_ingestion_config.test_data_path
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise IndiaInsuranceException(e,sys)
        

# if __name__ == "__main__":
#     obj = DataIngestion()
#     train_data,test_data = obj.initiate_data_ingestion()

#     data_transformation = DataTransformation()
#     train_ar,test_ar,_ = data_transformation.initiate_data_transformation(train_data,test_data)

#     modeltrainer = ModelTrainer()
#     print(modeltrainer.initiate_model_trainer(train_ar,test_ar))



    
