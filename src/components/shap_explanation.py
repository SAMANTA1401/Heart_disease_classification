import shap
import pandas as pd
import numpy as np
from src.path_config import ShapexplainerConfig, ModelTrainerConfig, DataIngestionConfig, DataTransformationConfig
from src.exception import CustomException
from src.logger import logging
import sys
import joblib


class Shap_explanation:
    def __init__(self):
        self.shapconfig = ShapexplainerConfig()

    def explain(self, model, X_train_scaled, X_test_scaled):
        try:
            # Convert DataFrame to numpy array to avoid catboost.Pool issue
            if hasattr(X_test_scaled, "values"):
                X_test_input = X_test_scaled.values
            else:
                X_test_input = X_test_scaled

            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)

            # Compute SHAP values
            shap_values = explainer(X_test_input)

            # Build SHAP DataFrame
            shap_df = pd.DataFrame(
                shap_values.values,
                columns=[f"Feature_{i}" for i in range(X_test_input.shape[1])]
            )

            # Original feature values DataFrame
            features_df = pd.DataFrame(
                X_test_input,
                columns=[f"Feature_{i}" for i in range(X_test_input.shape[1])]
            )

            # Combine both
            combined_df = pd.concat([features_df, shap_df.add_prefix("SHAP_")], axis=1)

            # Save to CSV
            combined_df.to_csv(self.shapconfig.shapdataexplainer, index=False)

            logging.info(f"SHAP explanation saved to {self.shapconfig.shapdataexplainer}")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # Load model
        model = joblib.load(ModelTrainerConfig.trained_model_file_path)

        # Load data
        X_train = pd.read_csv(DataTransformationConfig.preprocess_train_arr_path)
        X_test = pd.read_csv(DataTransformationConfig.preprocess_test_arr_path)

        # Drop target column if present
        # if 'HeartDisease' in X_train.columns:
        #     X_train = X_train.drop(columns=['HeartDisease'])
        # if 'HeartDisease' in X_test.columns:
        #     X_test = X_test.drop(columns=['HeartDisease'])

        # Run SHAP explanation
        shap_explainer = Shap_explanation()
        shap_explainer.explain(model, X_train, X_test)

    except Exception as e:
        raise CustomException(e, sys)
