import pandas as pd
import os
import numpy as np
from modAL.uncertainty import uncertainty_sampling
from utils.data_loader import load_data
from utils.model_utils import create_active_learner, save_model, load_model

def initialize_active_learner():
    """Initialize or load active learner with complete error handling"""
    try:
        # Ensure directories exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Try to load existing model
        if os.path.exists('models/active_learner.pkl'):
            learner = load_model('models/active_learner.pkl')
            
            # Load or initialize feedback data
            if os.path.exists('data/human_feedback.csv'):
                feedback_df = pd.read_csv('data/human_feedback.csv')
                # Validate feedback structure
                required_cols = {'text', 'model_pred', 'model_confidence', 'human_label'}
                if not required_cols.issubset(feedback_df.columns):
                    raise ValueError("Invalid feedback file structure")
            else:
                feedback_df = create_new_feedback_df()
                
            print("Loaded existing active learner")
            return learner, feedback_df
        
        # Create new learner if none exists
        X_train_vec, _, y_train, _, _, original_texts = load_data()
        
        # Validate pool sizes
        if X_train_vec.shape[0] != len(original_texts):
            raise ValueError("Feature and text pool size mismatch")
        
        learner = create_active_learner(
            X_train_vec[:100],
            y_train[:100]
        )
        
        feedback_df = create_new_feedback_df()
        save_active_learner(learner)
        save_feedback(feedback_df)
        
        print("Created new active learner")
        return learner, feedback_df
        
    except Exception as e:
        print(f"Active learner initialization failed: {str(e)}")
        raise

def create_new_feedback_df():
    """Create properly structured empty feedback DataFrame"""
    return pd.DataFrame(columns=[
        'text',         # Clean text string
        'model_pred',   # Model's prediction (0 or 1)
        'model_confidence',  # Confidence score (0-1)
        'human_label'   # Human correction (0 or 1)
    ])

def save_feedback(feedback_df):
    """Save feedback data with validation"""
    try:
        # Clean text data before saving
        if 'text' in feedback_df.columns:
            feedback_df['text'] = feedback_df['text'].astype(str).str.strip()
            
        # Validate structure
        required_cols = {'text', 'model_pred', 'model_confidence', 'human_label'}
        if not required_cols.issubset(feedback_df.columns):
            raise ValueError("Feedback data missing required columns")
            
        feedback_df.to_csv('data/human_feedback.csv', index=False)
    except Exception as e:
        print(f"Feedback save failed: {str(e)}")
        raise

def save_active_learner(learner):
    """Save learner with validation"""
    try:
        if not hasattr(learner, 'teach'):
            raise ValueError("Invalid learner object")
        save_model(learner, 'models/active_learner.pkl')
    except Exception as e:
        print(f"Learner save failed: {str(e)}")
        raise