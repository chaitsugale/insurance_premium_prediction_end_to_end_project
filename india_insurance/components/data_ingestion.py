import os
import sys
from india_insurance.exception import IndiaInsuranceException
from india_insurance.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass #defining variable

from india_insurance.components.data_transformation import DataTransformation

#any data input is required it will provided by data ingestion config class
@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts',"train.csv")
    test_data_path : str = os.path.join('artifacts',"test.csv")
    raw_data_path : str = os.path.join('artifacts',"raw.csv")

#other function inside the class then we can give init
class DataIngestion:
    def __init__(self):
        #3 files will get saved in this class variable
        self.ingestion_config = DataIngestionConfig()
    
    #reading the dataset, read in db then create mongodb client 
    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or components')
        try:
            df = pd.read_csv('notebook\insurance-premium-prediction\insurance.csv')

            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index = False,header = True)

            logging.info("Train test Split initiated")
            train_set,test_set = train_test_split(df,test_size = 0.3,random_state= 42)

            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False,header = True)

            logging.info('Ingestion of data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise IndiaInsuranceException(e,sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)

    
