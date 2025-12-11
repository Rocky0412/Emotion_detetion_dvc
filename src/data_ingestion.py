import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import requests
import pandas as pd
from io import StringIO
import yaml

with open('params.yaml','r') as f:
    parameter=yaml.safe_load(f)

def load_data(url:str):

    r = requests.get(url)
    data_url = StringIO(r.text)
    df = pd.read_csv(data_url)

    df.drop(columns=['tweet_id'],inplace=True)
    final_df=df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
    final_df['sentiment'] = final_df['sentiment'].map({'happiness': 1, 'sadness': 0})
    final_df = final_df.infer_objects(copy=False)

    train_data, test_data = train_test_split(final_df, test_size=parameter['data_ingestion']['test_size'], random_state=42)
    data_path=os.path.join("data","raw")
    os.makedirs(data_path,exist_ok=True)

    train_data.to_csv(os.path.join(data_path,'train.csv'))
    test_data.to_csv(os.path.join(data_path,'test.csv'))
    



data_url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
load_data(data_url)


