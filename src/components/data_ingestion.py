import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass


## Initialize the data ingestion configuration

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts", "train.csv") # It will be create a folder and file name
    test_data_path:str = os.path.join("artifacts", "test.csv")
    raw_data_path:str = os.path.join("artifacts", "raw.csv")

## Creating a class for data ingestion

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Method Started")
        
        try:
            df = pd.read_csv(os.path.join("notebooks/data/airline.csv"))
            logging.info("Dataset read as pandas dataframe")

            logging.info("Replacing Space in Columns Name's")

            df.columns = [i.replace(" ", "_") for i in df.columns]
            df.columns = [i.replace("/", "_or_") for i in df.columns]
            df.columns = [i.replace("-", "_")for i in df.columns]

            logging.info("Replaced Space in Columns Name's is completed")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("Train and Test Split Started")

            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            logging.info(f"Train Dataframe Sample's: \n{train_set.sample(2).to_string()}")
            logging.info(f"Test Dataframe Sample's: \n{test_set.sample(2).to_string()}")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception Occured at Data Ingestion Stage")
            raise CustomException(e,sys)
        
## Run data ingestion

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)