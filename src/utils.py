import os 
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import f1_score , accuracy_score, precision_recall_curve
from sklearn.preprocessing import OrdinalEncoder 


# 12 after data transformation then go data ingestion
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise logging.info(CustomException(e,sys))
    

def feature_engineering(df):
    # Binning Age (continuous variable to categorical)
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 40, 60, 100], labels=['Young', 'Middle-aged', 'Old'])

    # Binary transform for high cholesterol
    df['HighCholesterol'] = (df['Cholesterol'] > 240).astype(int)

    # Combine Oldpeak & MaxHR into a StressScore
    df['StressScore'] = df['Oldpeak'] * (220 - df['MaxHR']) / 220

    # Interaction features
    df['BPxChol'] = df['RestingBP'] * df['Cholesterol']
    df['FBSxOldpeak'] = df['FastingBS'] * df['Oldpeak']

    # Drop columns that are now redundant if needed
    # df = df.drop(columns=['Oldpeak', 'Cholesterol', 'RestingBP'])

    return df