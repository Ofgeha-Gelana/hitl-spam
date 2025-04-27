import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from modAL.models import ActiveLearner

def create_baseline_model(X_train, y_train):
    """Create and train a baseline model with validation"""
    try:
        model = MultinomialNB()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error creating baseline model: {str(e)}")
        raise

def create_active_learner(X_train, y_train):
    """Initialize active learner with proper data types and validation"""
    try:
        # Convert to appropriate formats
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()  # Convert sparse to dense
        y_train = np.array(y_train, dtype=int)
        
        # Validate shapes
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X and y have mismatched sample counts")
            
        # Create and return learner
        return ActiveLearner(
            estimator=MultinomialNB(),
            X_training=X_train,
            y_training=y_train
        )
    except Exception as e:
        print(f"Error creating active learner: {str(e)}")
        raise

def save_model(model, path):
    """Save model to file with error handling"""
    try:
        joblib.dump(model, path)
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        raise

def load_model(path):
    """Load model from file with error handling"""
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise