
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
import pandas as pd
from src.path_config import DataIngestionConfig



class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        pass

    def data_splitting(self):
        try:
           df = pd.read_csv(self.ingestion_config.raw_data_path)

           train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

           train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

           test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

           return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise logging.info(CustomException(e,sys))

if __name__=="__main__":
    obj = DataIngestion()
    train_data,test_data=obj.data_splitting()