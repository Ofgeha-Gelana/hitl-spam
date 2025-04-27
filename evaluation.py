import pandas as pd
import matplotlib.pyplot as plt
from utils.data_loader import load_data
from utils.model_utils import load_model
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, confusion_matrix)
import seaborn as sns
import os




def evaluate_models():
    """Evaluate and compare baseline vs HITL model"""
    # Load data and models
    X_train, X_test, y_train, y_test, _ = load_data()
    baseline = load_model('models/baseline_model.pkl')
    hitl = load_model('models/active_learner.pkl')
    
    # Make predictions
    base_preds = baseline.predict(X_test)
    hitl_preds = hitl.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'Model': ['Baseline', 'HITL'],
        'Accuracy': [
            accuracy_score(y_test, base_preds),
            accuracy_score(y_test, hitl_preds)
        ],
        'Precision': [
            precision_score(y_test, base_preds),
            precision_score(y_test, hitl_preds)
        ],
        'Recall': [
            recall_score(y_test, base_preds),
            recall_score(y_test, hitl_preds)
        ],
        'F1': [
            f1_score(y_test, base_preds),
            f1_score(y_test, hitl_preds)
        ]
    }
    
    # Create visualizations
    os.makedirs('results', exist_ok=True)
    
    # 1. Metrics comparison
    df_metrics = pd.DataFrame(metrics)
    plt.figure(figsize=(10, 6))
    df_metrics.set_index('Model').plot(kind='bar', rot=0)
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig('results/metrics_comparison.png')
    plt.close()
    
    # 2. Confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(confusion_matrix(y_test, base_preds), 
                annot=True, fmt='d', ax=ax1, cmap='Blues')
    ax1.set_title('Baseline Model')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    sns.heatmap(confusion_matrix(y_test, hitl_preds), 
                annot=True, fmt='d', ax=ax2, cmap='Blues')
    ax2.set_title('HITL Model')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrices.png')
    plt.close()
    
    # 3. Learning curve (if feedback exists)
    if os.path.exists('data/human_feedback.csv'):
        feedback = pd.read_csv('data/human_feedback.csv')
        if len(feedback) > 0:
            plt.figure(figsize=(10, 6))
            feedback['model_confidence'].plot()
            plt.title('Model Confidence Over Feedback Iterations')
            plt.xlabel('Feedback Samples')
            plt.ylabel('Confidence Score')
            plt.savefig('results/learning_curve.png')
            plt.close()
    
    return df_metrics

if __name__ == "__main__":
    metrics = evaluate_models()
    print(metrics)
    print("\nEvaluation results saved to /results directory")