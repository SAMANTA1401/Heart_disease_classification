import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.path_config import DataTransformationConfig
# 10. after data ingestion

# 11.
class DataTransformation:
    # 11.1
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformer_object(self):
        ''' this function is responsible for data transformation '''
        try:
            numerical_columns = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak','HighCholesterol', 'StressScore', 'BPxChol','FBSxOldpeak']
            categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope','AgeBin']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                    ]

                )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder(handle_unknown='ignore')),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise logging.info(CustomException(e,sys))
        
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
        

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name="HeartDisease"
            

            input_feature_train__df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test__df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]


            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train__df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test__df)

       
            np.savetxt(self.data_transformation_config.preprocess_train_arr_path, input_feature_train_arr, delimiter=",", fmt='%f')
            np.savetxt(self.data_transformation_config.preprocess_test_arr_path, input_feature_test_arr, delimiter=",", fmt='%f')

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr =  np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

        

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise logging.info(CustomException(e,sys))