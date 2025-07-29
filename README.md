# 🫀 Heart Failure Prediction Project

This project aims to predict the likelihood of heart failure using machine learning techniques. It uses a dataset from Kaggle and provides model training, evaluation, and SHAP-based explainability. Additionally, a Streamlit web interface is provided for interactive predictions.

---

##  Data Source

The dataset is sourced from Kaggle:

**Heart Failure Prediction Dataset**  
🔗 [https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction?resource=download)

---

## Environment Setup

Make sure you have **Conda** installed. Follow the steps below to create and activate a virtual environment:

```bash
# Create a virtual environment named 'venv' in the project directory
conda create -p venv python==<version> -y

# Activate the environment (Linux/Mac/Windows with Conda)
conda activate ./venv/

# Install project dependencies
pip install -r requirements.txt
```

## To train the machine learning pipeline, run the following script:
```bash
python src/pipeline/train_pipeline.py
```

## The Streamlit app allows users to make predictions using the trained model. Run the app using:
```bash
streamlit run app.py
```
