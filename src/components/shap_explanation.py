import shap
import pandas as pd
from src.path_config import ShapexplainerConfig
from src.path_config import ModelTrainerConfig
from src.path_config import DataIngestionConfig
from src.exception import CustomException
from src.logger import logging
import sys


class Shap_explanation:
    def __init__(self):
        self.shapconfig = ShapexplainerConfig()
        

    def explain(self,model, X_train_scaled, X_test_scaled):

        try:

      
            explainer = shap.TreeExplainer(model)
         

            # Compute SHAP values
            shap_values = explainer(X_test_scaled)

            # Convert shap values and features to DataFrame
            shap_df = pd.DataFrame(shap_values.values, columns=[f"Feature_{i}" for i in range(X_test_scaled.shape[1])])
            features_df = pd.DataFrame(X_test_scaled, columns=[f"Feature_{i}" for i in range(X_test_scaled.shape[1])])

            # Combine features and SHAP values for better traceability
            combined_df = pd.concat([features_df, shap_df.add_prefix("SHAP_")], axis=1)

            # Save to CSV
            combined_df.to_csv(self.shapconfig.shapdataexplainer, index=False)
        
        except Exception as e:
            raise logging.info(CustomException(e,sys))


if __name__ == "__main__":
    import joblib
    model = joblib.load(ModelTrainerConfig.trained_model_file_path)
    shap_explainer = Shap_explanation()
    shap_explainer.explain(model,DataIngestionConfig.train_data_path,DataIngestionConfig.test_data_path)