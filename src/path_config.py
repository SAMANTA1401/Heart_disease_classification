import os
from dataclasses import dataclass




@dataclass
class DataIngestionConfig:  
    raw_data_path: str=os.path.join('data','balance_heart_data.csv') #save raw dataset in this path
    train_data_path: str=os.path.join('artifacts','train.csv') #save train data in this path
    test_data_path:str = os.path.join('artifacts', 'test.csv')   # save test data in this path

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

@dataclass
class ShapexplainerConfig:
    shapdataexplainer=os.path.join('model_evaluation',"shap_values_with_features.csv")

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("model","best_model.joblib")
    model_evaluation_file_path = os.path.join("model_evaluation","evaluation_report.csv")

