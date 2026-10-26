# Telco Customer Churn Prediction

An end-to-end supervised machine learning project that predicts customer attrition and evaluates retention risk across 7,000+ telecom customer profiles. Built with Python, Scikit-Learn, XGBoost, and Streamlit.

## Overview
Customer churn is a critical metric in subscription-based businesses. This project implements a complete data preprocessing, feature engineering, and model evaluation pipeline on the IBM Telco Customer Churn dataset, comparing linear models, tree-based ensembles, and neural networks.

## Key Features
- **Data Cleaning & Feature Engineering:** Handled missing billing records, engineered a `Charge_to_Tenure_Ratio` feature, one-hot encoded categorical variables, and standardized continuous features using `StandardScaler`.
- **Multi-Model Benchmarking:** Trained and evaluated 5 architectures using 5-Fold Stratified Cross-Validation:
  - Logistic Regression (L2 Regularized)
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - XGBoost Classifier
  - Artificial Neural Network (`MLPClassifier` with ReLU activation and Early Stopping)
- **Interactive Analytics & Prediction App:** Built a Streamlit dashboard featuring live ROC-AUC curves, confusion matrices, EDA visualizations, and an interactive single-customer churn risk calculator.

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn
- **Web Framework:** Streamlit

## Project Structure
```text
├── app.py                              # Streamlit web application and model training pipeline
├── churn_analysis_and_modeling.ipynb   # Jupyter Notebook for EDA and model experimentation
├── requirements.txt                    # Python package dependencies
└── README.md                           # Project documentation
