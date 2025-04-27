import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from modAL.models import ActiveLearner

def create_baseline_model(X_train, y_train):
    """Create baseline model with validation"""
    try:
        if X_train.shape[0] != len(y_train):
            raise ValueError("X and y have mismatched lengths")
            
        model = MultinomialNB()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Baseline model creation failed: {str(e)}")
        raise

def create_active_learner(X_train, y_train):
    """Initialize active learner with proper data validation"""
    try:
        # Convert sparse matrices if needed
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        y_train = np.array(y_train, dtype=int)
        
        # Validate shapes
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Training data/labels mismatch")
            
        return ActiveLearner(
            estimator=MultinomialNB(),
            X_training=X_train,
            y_training=y_train
        )
    except Exception as e:
        print(f"Active learner creation failed: {str(e)}")
        raise

def save_model(model, path):
    """Save model with error handling"""
    try:
        joblib.dump(model, path)
    except Exception as e:
        print(f"Model save failed: {str(e)}")
        raise

def load_model(path):
    """Load model with error handling"""
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Model load failed: {str(e)}")
        raise