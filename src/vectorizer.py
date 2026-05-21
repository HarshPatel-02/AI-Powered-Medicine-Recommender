import pickle 
import os
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import VECTORIZER_PATH


def create_vectorizer(df,force_recreate=False):
    if os.path.exists(VECTORIZER_PATH) and not force_recreate:
        with open(VECTORIZER_PATH,'rb')as f:
            vectorizer = pickle.load(f)
            print("TF-IDF Vectorizer loaded from file successfully")
    else:
        print("Creating new TF-IDF Vectorizer")
        vectorizer = TfidfVectorizer(
           stop_words='english',
            ngram_range=(1,2)
        )
        vectorizer.fit(df['combined_text'])

        #save vectorizer to file
        os.makedirs(os.path.dirname(VECTORIZER_PATH),exist_ok=True)
        with open(VECTORIZER_PATH,'wb') as f:
            pickle.dump(vectorizer,f)
        print("TF-IDF Vectorizer trained and saved to file successfully")

    return vectorizer
