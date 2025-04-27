import pandas as pd
import os
import numpy as np
from modAL.uncertainty import uncertainty_sampling
from utils.data_loader import load_data
from utils.model_utils import create_active_learner, save_model, load_model

def initialize_active_learner():
    """Initialize or load active learner with comprehensive error handling"""
    try:
        # Ensure directories exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Try to load existing model
        if os.path.exists('models/active_learner.pkl'):
            learner = load_model('models/active_learner.pkl')
            
            # Load or create feedback DataFrame
            if os.path.exists('data/human_feedback.csv'):
                feedback_df = pd.read_csv('data/human_feedback.csv')
                # Validate feedback data structure
                required_cols = {'text', 'model_pred', 'model_confidence', 'human_label'}
                if not required_cols.issubset(feedback_df.columns):
                    raise ValueError("Feedback file missing required columns")
            else:
                feedback_df = create_new_feedback_df()
                
            print("Loaded existing active learner")
            return learner, feedback_df
        
        # Create new learner if none exists
        X_train_vec, _, y_train, _, _, _ = load_data()
        
        # Initialize with first 100 samples
        learner = create_active_learner(
            X_train_vec[:100],
            y_train[:100]
        )
        
        # Create new feedback DataFrame
        feedback_df = create_new_feedback_df()
        
        # Save initial state
        save_active_learner(learner)
        save_feedback(feedback_df)
        
        print("Created new active learner")
        return learner, feedback_df
        
    except Exception as e:
        print(f"Failed to initialize active learner: {str(e)}")
        raise

def create_new_feedback_df():
    """Create a new empty feedback DataFrame"""
    return pd.DataFrame(columns=[
        'text', 'model_pred', 'model_confidence', 'human_label'
    ])

def save_feedback(feedback_df):
    """Save feedback data with validation"""
    try:
        required_cols = {'text', 'model_pred', 'model_confidence', 'human_label'}
        if not required_cols.issubset(feedback_df.columns):
            raise ValueError("Feedback DataFrame missing required columns")
            
        feedback_df.to_csv('data/human_feedback.csv', index=False)
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")
        raise

def save_active_learner(learner):
    """Save active learner model with validation"""
    try:
        if not hasattr(learner, 'teach'):
            raise ValueError("Invalid learner object")
        save_model(learner, 'models/active_learner.pkl')
    except Exception as e:
        print(f"Error saving active learner: {str(e)}")
        raise