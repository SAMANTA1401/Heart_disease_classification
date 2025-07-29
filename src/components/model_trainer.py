
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import joblib
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.metrics import f1_score , accuracy_score, precision_score, recall_score
from src.path_config import ModelTrainerConfig




class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    ## 15.2
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], 
                train_array[:,-1],  
                test_array[:, :-1],  
                test_array[:, -1]

            )

            models_params = {
                "LogisticRegression": (
                    LogisticRegression(max_iter=1000, random_state=42),
                    {
                        "C": [0.01, 0.1, 1, 10],
                    }
                ),
                "RandomForest": (
                    RandomForestClassifier(random_state=42),
                    {
                        "n_estimators": [100, 200],
                        "max_depth": [5, 10, None]
                    }
                ),
                "XGBoost": (
                    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                    {
                        "n_estimators": [100, 200],
                        "max_depth": [3, 6]
                    }
                ),
                "CatBoost": (
                    CatBoostClassifier(verbose=0, random_state=42),
                    {
                        "depth": [4, 6],
                        "learning_rate": [0.01, 0.1]
                    }
                )
                
            }

            results = []
            best_model = None
            best_score = 0
            best_model_name = None
            best_params = None

            for name, (model, params) in models_params.items():

                grid = GridSearchCV(model, params, cv=5, scoring='f1', n_jobs=-1, verbose=0)
                grid.fit(X_train, y_train)

                y_pred = grid.predict(X_test)

                try:
                    y_prob = grid.predict_proba(X_test)
                    auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
                except:
                    auc = None

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')

                results.append({
                    "Model": name,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1 Score": f1,
                    "auc": auc,
                    "Best Params": grid.best_params_
                })

                if f1 > best_score:
                    best_score = f1
                    best_model = grid.best_estimator_
                    best_model_name = name
                    best_params = grid.best_params_


            df_results = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)

            df_results.to_csv(self.model_trainer_config.model_evaluation_file_path)

            joblib.dump(best_model, self.model_trainer_config.trained_model_file_path)

        except Exception as e:
            raise logging.info(CustomException(e,sys))