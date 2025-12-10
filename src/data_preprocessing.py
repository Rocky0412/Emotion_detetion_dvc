import numpy as np
import pandas as pd
import os
import re
import nltk
import string
import ssl

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# -------- FIX SSL ISSUE FOR MAC ----------
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except:
    pass

# -------- DOWNLOAD NLTK DATA -------------
nltk.download('stopwords')
nltk.download('wordnet')

# -------- INITIALIZE GLOBALS -------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def lematization(text):
    words = text.split()
    return " ".join(lemmatizer.lemmatize(w) for w in words)

def remove_stopword(text):
    words = text.split()
    return " ".join(w for w in words if w not in stop_words)

def remove_digit(text):
    words = text.split()
    return " ".join(w for w in words if not w.isdigit())

def lower_case(text):
    return text.lower()

def removing_punctuations(text):
    text = re.sub(r'[!"#$%&\'()*+,./:;<=>?@[\]^_`{|}~]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def normalize_text(df):
    df['content'] = df['content'].apply(lower_case)\
                                 .apply(remove_stopword)\
                                 .apply(remove_digit)\
                                 .apply(removing_punctuations)\
                                 .apply(removing_urls)\
                                 .apply(lematization)
    return df


def datapreprocessing(path:str, file:str):
    file_path = os.path.join(path, file)
    df = pd.read_csv(file_path)
    return normalize_text(df)


def main():
    raw_path = 'data/raw'
    processed_path = 'data/processed'

    # Create processed folder
    os.makedirs(processed_path, exist_ok=True)

    test_data = datapreprocessing(raw_path, 'test.csv')
    train_data = datapreprocessing(raw_path, 'train.csv')

    test_data.to_csv(os.path.join(processed_path, 'test.csv'), index=False)
    train_data.to_csv(os.path.join(processed_path, 'train.csv'), index=False)


if __name__ == "__main__":
    main()
