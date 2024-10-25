# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:45:20 2024

@author: JINGYI HUO
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the classifiers
gbdt = GradientBoostingClassifier(random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Train the models
gbdt.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Make predictions
y_pred_gbdt = gbdt.predict(X_test)
y_pred_xgb = xgb.predict(X_test)

# Define a function to calculate evaluation metrics
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Roc_Auc_Score: {auc:.4f}")
    
# Evaluate both models
evaluate_model(y_test, y_pred_gbdt, "Gradient Boosting (GBDT)")
evaluate_model(y_test, y_pred_xgb, "XGBoost")