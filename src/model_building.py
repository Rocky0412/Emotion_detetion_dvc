import numpy as np
import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import yaml

#load parameter yaml
with open('params.yaml','r') as f:
    parameter=yaml.safe_load(f)

#Parameters

nestimator=parameter['model_building']['n_estimator']
learning_rate=parameter['model_building']['learning_rate']

# Load dataset
df = pd.read_csv('./data/features/train_bow.csv')

# Prepare features and label
xtrain = df.iloc[:, 0:-1].values
ytrain = df.iloc[:, -1].values

# Train Model
gbmodel = GradientBoostingClassifier(n_estimators=nestimator,learning_rate=learning_rate)
gbmodel.fit(xtrain, ytrain)

# Save model
model_path = "models/emotion_model.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)

with open(model_path, "wb") as f:
    pickle.dump(gbmodel, f)

print("Model saved successfully!")


