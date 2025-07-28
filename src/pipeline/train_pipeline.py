from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import sys



class TrainPipeline:
    def __init__(self):
        self.ingest = DataIngestion()
        self.transform = DataTransformation()
        self.training = ModelTrainer()

        pass 

    def train(self):
        try:

            train_path, test_path = self.ingest.data_splitting()

            train_arr, test_arr, preprocess_obj = self.transform.initiate_data_transformation(train_path,test_path)

            self.training.initiate_model_trainer(train_arr,test_arr)

            logging.info("training done")


        except Exception as e:
            raise logging.info(CustomException(e,sys))
        


if __name__ == "__main__":
    train = TrainPipeline().train()


        
