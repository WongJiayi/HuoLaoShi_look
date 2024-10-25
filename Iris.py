# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:32:21 2024

@author: JINGYI HUO
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Binarize the labels for AUC calculation (one-hot encoding)
y_binarized = label_binarize(y, classes=[0, 1, 2])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])

# Initialize the AdaBoost classifier
adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)

# Train the model
adaboost.fit(X_train, y_train)

# Make predictions
y_pred = adaboost.predict(X_test)
y_pred_proba = adaboost.predict_proba(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
auc = roc_auc_score(y_test_binarized, y_pred_proba, average='weighted', multi_class='ovr')

print(f"AdaBoost Model Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"AUC: {auc:.2f}")
