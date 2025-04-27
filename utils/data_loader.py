import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data():
    """Load and preprocess the SMS spam dataset with robust error handling"""
    try:
        # Load data with explicit encoding
        df = pd.read_csv('data/spam.csv', encoding='latin-1')
        
        # Clean and validate data
        required_columns = ['v1', 'v2']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Data file missing required columns")
            
        df = df[required_columns].rename(columns={'v1': 'label', 'v2': 'text'})
        df = df.dropna()
        df = df[df['text'].notna() & (df['text'].str.strip() != '')]
        
        # Verify we have enough data
        if len(df) < 100:
            raise ValueError("Insufficient data samples (need at least 100)")
        
        # Convert labels
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        df['label'] = df['label'].astype(int)
        
        # Split data (keeping texts as strings)
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'].astype(str),  # Ensure strings
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
                X_train.reset_index(drop=True))  # Return original texts as Series
                
    except Exception as e:
        print(f"CRITICAL ERROR loading data: {str(e)}")
        raise