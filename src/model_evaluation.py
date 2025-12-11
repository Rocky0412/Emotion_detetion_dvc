import pickle
import numpy as np
import pandas as pd
import yaml
import os
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    recall_score,
    roc_auc_score
)

# Load model
model_path = './models/emotion_model.pkl'
data_path = './data/features/test_bow.csv'

with open(model_path, 'rb') as f:
    gbclassifier = pickle.load(f)

# Load test data
test_df = pd.read_csv(data_path)

X_test = test_df.iloc[:, 0:-1]
y_test = test_df.iloc[:, -1]

# Predictions
y_pred = gbclassifier.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, gbclassifier.predict_proba(X_test)[:, 1])

metrics = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'roc_auc_score': float(roc_auc)
}

# Save metrics
metrics_path = os.path.join('evaluation', 'metrics.yaml')
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

with open(metrics_path, 'w') as f:
    yaml.dump(metrics, f)

print("Metrics saved to", metrics_path)


