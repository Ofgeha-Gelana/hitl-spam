import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from utils.data_loader import load_data
from utils.model_utils import load_model
import os

def evaluate_models():
    """Evaluate and compare baseline vs HITL model performance"""
    try:
        # Load data and models
        X_train_vec, X_test_vec, y_train, y_test, vectorizer, _ = load_data()
        baseline = load_model('models/baseline_model.pkl')
        hitl = load_model('models/active_learner.pkl')
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # 1. Generate predictions
        base_preds = baseline.predict(X_test_vec)
        hitl_preds = hitl.predict(X_test_vec)
        
        # Handle case where no positive predictions exist
        pos_label = 1  # Assuming 1 is the positive class (spam)
        zero_division = 0  # Value to return when there's a zero division
        
        # 2. Calculate metrics
        metrics = {
            'Model': ['Baseline', 'HITL'],
            'Accuracy': [
                accuracy_score(y_test, base_preds),
                accuracy_score(y_test, hitl_preds)
            ],
            'Precision': [
                precision_score(y_test, base_preds, pos_label=pos_label, zero_division=zero_division),
                precision_score(y_test, hitl_preds, pos_label=pos_label, zero_division=zero_division)
            ],
            'Recall': [
                recall_score(y_test, base_preds, pos_label=pos_label, zero_division=zero_division),
                recall_score(y_test, hitl_preds, pos_label=pos_label, zero_division=zero_division)
            ],
            'F1-Score': [
                f1_score(y_test, base_preds, pos_label=pos_label, zero_division=zero_division),
                f1_score(y_test, hitl_preds, pos_label=pos_label, zero_division=zero_division)
            ]
        }
        metrics_df = pd.DataFrame(metrics)
        
        # 3. Generate visualizations
        sns.set_style("whitegrid")  # Use Seaborn's style
        
        # Metrics comparison bar plot
        plt.figure(figsize=(10, 6))
        ax = metrics_df.set_index('Model').plot(kind='bar', rot=0)
        plt.title('Model Performance Comparison', pad=20)
        plt.ylabel('Score')
        plt.ylim(0, 1.05)
        
        # Add values on top of bars
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig('results/metrics_comparison.png')
        plt.close()
        
        # Confusion matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        sns.heatmap(
            confusion_matrix(y_test, base_preds),
            annot=True, fmt='d', ax=ax1, cmap='Blues',
            cbar=False
        )
        ax1.set_title('Baseline Model', pad=20)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        sns.heatmap(
            confusion_matrix(y_test, hitl_preds),
            annot=True, fmt='d', ax=ax2, cmap='Blues',
            cbar=False
        )
        ax2.set_title('HITL Model', pad=20)
        ax2.set_xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('results/confusion_matrices.png')
        plt.close()
        
        # ROC Curve (only if probabilities are available)
        try:
            base_probs = baseline.predict_proba(X_test_vec)[:, 1]
            hitl_probs = hitl.predict_proba(X_test_vec)[:, 1]
            
            base_fpr, base_tpr, _ = roc_curve(y_test, base_probs)
            hitl_fpr, hitl_tpr, _ = roc_curve(y_test, hitl_probs)
            
            plt.figure(figsize=(8, 6))
            plt.plot(base_fpr, base_tpr, label=f'Baseline (AUC = {auc(base_fpr, base_tpr):.2f})')
            plt.plot(hitl_fpr, hitl_tpr, label=f'HITL (AUC = {auc(hitl_fpr, hitl_tpr):.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Comparison', pad=20)
            plt.legend()
            plt.savefig('results/roc_curve.png')
            plt.close()
        except AttributeError:
            print("ROC curve skipped - model doesn't support predict_proba")
        
        # 4. Save detailed reports
        with open('results/classification_reports.txt', 'w') as f:
            f.write("=== BASELINE MODEL ===\n")
            f.write(classification_report(y_test, base_preds, zero_division=zero_division))
            f.write("\n=== HITL MODEL ===\n")
            f.write(classification_report(y_test, hitl_preds, zero_division=zero_division))
        
        metrics_df.to_csv('results/metrics.csv', index=False)
        
        print("✅ Evaluation complete! Results saved to /results directory")
        print("\nModel Metrics Summary:")
        print(metrics_df.to_markdown(index=False, tablefmt="grid"))
        
        return metrics_df
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    results = evaluate_models()