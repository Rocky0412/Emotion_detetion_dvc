import numpy as np
import pandas as pd
import os 
from sklearn.feature_extraction.text import CountVectorizer

def feature_engineering(path:str):
    test_path=os.path.join(path,'test.csv')
    train_path=os.path.join(path,'train.csv')
    test_df=pd.read_csv(test_path)
    train_df=pd.read_csv(train_path)
    vectorizer=CountVectorizer(max_features=100)
    #apply bag of words
    X_train=vectorizer.fit_transform(train_df['content'].fillna("").values)
    X_test=vectorizer.transform(test_df['content'].fillna("").values)

    X_train=pd.DataFrame(X_train.toarray())
    X_test=pd.DataFrame(X_test.toarray())

    X_train['sentiment']=train_df['sentiment']
    X_test['sentiment']=test_df['sentiment']

    stored_path=os.path.join('data','features')

    os.makedirs(stored_path,exist_ok=True)

    X_train.to_csv(os.path.join(stored_path,'train_bow.csv'))
    X_test.to_csv(os.path.join(stored_path,'test_bow.csv'))


def main():
    path='./data/processed'
    feature_engineering(path)



if __name__=="__main__":
    main()


    
    