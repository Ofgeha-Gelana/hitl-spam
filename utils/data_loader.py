import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data():
    """Load and preprocess the SMS spam dataset with robust validation"""
    try:
        # Load and validate data
        df = pd.read_csv('data/spam.csv', encoding='latin-1')
        
        # Check required columns
        if not all(col in df.columns for col in ['v1', 'v2']):
            raise ValueError("Dataset missing required columns 'v1' or 'v2'")
            
        df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
        df = df.dropna()
        df = df[df['text'].notna() & (df['text'].str.strip() != '')]
        
        # Convert labels and validate
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        if df['label'].isna().any():
            raise ValueError("Invalid labels found in dataset")
        df['label'] = df['label'].astype(int)
        
        # Split data (keeping original texts)
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'].astype(str),
            df['label'],
            test_size=0.2,
            random_state=42,
            stratify=df['label']
        )
        
        # Vectorize text
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        return (X_train_vec, X_test_vec,
                y_train.values, y_test.values,
                vectorizer,
                X_train.reset_index(drop=True))  # Return clean text Series
                
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        raise