import numpy as np
import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier
import pickle

# Load dataset
df = pd.read_csv('./data/features/train_bow.csv')

# Prepare features and label
xtrain = df.iloc[:, 0:-1].values
ytrain = df.iloc[:, -1].values

# Train Model
gbmodel = GradientBoostingClassifier()
gbmodel.fit(xtrain, ytrain)

# Save model
model_path = "models/emotion_model.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)

with open(model_path, "wb") as f:
    pickle.dump(gbmodel, f)

print("Model saved successfully!")


