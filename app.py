import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from modAL.uncertainty import uncertainty_sampling
from utils.data_loader import load_data
from utils.model_utils import load_model, create_active_learner
from active_learning import initialize_active_learner, save_feedback, save_active_learner

# Initialize directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)

def init_session():
    """Initialize session state with complete error handling"""
    try:
        if 'initialized' not in st.session_state:
            # Load data
            (X_train_vec, X_test_vec,
             y_train, y_test,
             vectorizer, original_texts) = load_data()
            
            # Validate pool sizes
            if X_train_vec.shape[0] != len(original_texts):
                raise ValueError("Feature and text pool size mismatch")
            
            # Store in session state
            st.session_state.update({
                'X_pool': X_train_vec,
                'y_pool': y_train,
                'vectorizer': vectorizer,
                'original_texts': original_texts,
                'remaining_texts': original_texts.copy(),
                'remaining_indices': np.arange(len(original_texts)),
                'initialized': True
            })
            
            # Initialize learner and feedback
            st.session_state.learner, st.session_state.feedback_df = initialize_active_learner()
            
    except Exception as e:
        st.error(f"Application initialization failed: {str(e)}")
        st.stop()

def get_new_sample():
    """Get a new sample with robust pool management"""
    try:
        # Check if pool is empty
        if len(st.session_state.remaining_indices) == 0:
            st.warning("No more samples available! Reset pool to continue.")
            return None
            
        # Get the active pool subset
        active_pool = st.session_state.X_pool[st.session_state.remaining_indices]
        
        # Find uncertain sample from active pool
        query_idx, _ = uncertainty_sampling(
            st.session_state.learner,
            active_pool
        )
        
        # Get the actual index in original pool
        actual_idx = st.session_state.remaining_indices[query_idx]
        
        # Get the text sample
        sample_text = str(st.session_state.original_texts.iloc[actual_idx])
        
        # Remove from remaining indices
        st.session_state.remaining_indices = np.delete(
            st.session_state.remaining_indices,
            query_idx
        )
        
        return sample_text.strip()
        
    except Exception as e:
        st.error(f"Sample retrieval error: {str(e)}")
        return None

def plot_confusion_matrix(model, X, y, ax, title):
    """Plot confusion matrix with error handling"""
    try:
        preds = model.predict(X.toarray())
        cm = confusion_matrix(y, preds)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    except Exception as e:
        ax.set_title(f"Error: {str(e)}")

def main():
    st.set_page_config(
        page_title="HITL Spam Classifier",
        page_icon="📧",
        layout="wide"
    )
    
    # Initialize session
    init_session()
    
    # Check initialization
    if not st.session_state.get('initialized', False):
        st.error("Application failed to initialize. Please check the data files.")
        st.stop()
    
    st.title("📧 Human-in-the-Loop Spam Classifier")
    st.markdown("""
        Review uncertain predictions to help improve the spam detection model.
        The system learns from your corrections!
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("📊 Dashboard")
        
        # Sample tracking
        total_samples = len(st.session_state.original_texts)
        remaining = len(st.session_state.remaining_indices)
        reviewed = total_samples - remaining
        st.metric("Total Samples", total_samples)
        st.metric("Remaining Samples", remaining)
        st.metric("Reviewed Samples", reviewed)
        
        # Feedback stats
        if not st.session_state.feedback_df.empty:
            corrections = sum(
                st.session_state.feedback_df['model_pred'] != 
                st.session_state.feedback_df['human_label']
            )
            st.metric("Corrections Made", corrections)
        
        # Pool reset
        if st.button("🔄 Reset Sample Pool"):
            try:
                # Reload the original data
                (X_train_vec, _, y_train, _, _, original_texts) = load_data()
                
                # Reset all pools
                st.session_state.X_pool = X_train_vec
                st.session_state.remaining_texts = original_texts.copy()
                st.session_state.remaining_indices = np.arange(len(original_texts))
                
                # Reinitialize the learner
                st.session_state.learner = create_active_learner(
                    X_train_vec[:100],
                    y_train[:100]
                )
                
                st.success("Sample pool and model reset successfully!")
            except Exception as e:
                st.error(f"Reset failed: {str(e)}")
    
    # Main interface tabs
    tab1, tab2 = st.tabs(["💬 Provide Feedback", "📈 Performance Analysis"])
    
    with tab1:
        st.header("Review Messages")
        
        # Get new sample
        if st.button("🎯 Get New Sample", type="primary"):
            st.session_state.current_text = get_new_sample()
            st.session_state.show_correction = False
        
        # Current sample display
        if hasattr(st.session_state, 'current_text') and st.session_state.current_text:
            with st.expander("📩 Current Message", expanded=True):
                st.write(st.session_state.current_text)
                
                # Get prediction
                try:
                    X_vec = st.session_state.vectorizer.transform(
                        [st.session_state.current_text]
                    )
                    pred = st.session_state.learner.predict(X_vec.toarray())[0]
                    proba = st.session_state.learner.predict_proba(X_vec.toarray())[0]
                    confidence = max(proba)
                    
                    st.markdown(f"""
                        **Model Prediction:**  
                        <span style="color: {'red' if pred == 1 else 'green'}; 
                                     font-weight: bold">
                        {'SPAM 🚨' if pred == 1 else 'HAM ✅'}
                        </span>  
                        **Confidence:** {confidence:.1%}
                    """, unsafe_allow_html=True)
                    
                    # Feedback options
                    cols = st.columns(3)
                    with cols[0]:
                        if st.button("👍 Correct", help="Confirm the prediction is accurate"):
                            new_feedback = {
                                'text': st.session_state.current_text,
                                'model_pred': pred,
                                'human_label': pred,
                                'model_confidence': confidence
                            }
                            st.session_state.feedback_df = pd.concat([
                                st.session_state.feedback_df,
                                pd.DataFrame([new_feedback])
                            ], ignore_index=True)
                            save_feedback(st.session_state.feedback_df)
                            st.success("Thanks for confirming!")
                            st.session_state.current_text = None
                    
                    with cols[1]:
                        if st.button("👎 Incorrect", help="Correct the model's mistake"):
                            st.session_state.show_correction = True
                    
                    with cols[2]:
                        if st.button("⏭ Skip", help="Skip this message"):
                            st.info("Message skipped")
                            st.session_state.current_text = None
                    
                    # Correction interface
                    if st.session_state.get('show_correction', False):
                        correction = st.radio(
                            "Correct label:",
                            ['HAM (not spam)', 'SPAM'],
                            horizontal=True
                        )
                        
                        if st.button("Submit Correction"):
                            corrected_label = 1 if 'SPAM' in correction else 0
                            
                            # Teach the model
                            st.session_state.learner.teach(
                                X_vec.toarray(),
                                np.array([corrected_label])
                            )
                            
                            # Save feedback
                            new_feedback = {
                                'text': st.session_state.current_text,
                                'model_pred': pred,
                                'human_label': corrected_label,
                                'model_confidence': confidence
                            }
                            st.session_state.feedback_df = pd.concat([
                                st.session_state.feedback_df,
                                pd.DataFrame([new_feedback])
                            ], ignore_index=True)
                            save_feedback(st.session_state.feedback_df)
                            save_active_learner(st.session_state.learner)
                            st.success("Model updated with your correction!")
                            st.session_state.show_correction = False
                            st.session_state.current_text = None
                
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    
    with tab2:
        st.header("Performance Analysis")
        
        try:
            _, X_test, _, y_test, _, _ = load_data()
            baseline = load_model('models/baseline_model.pkl')
            hitl = st.session_state.learner
            
            # Calculate metrics
            base_acc = accuracy_score(y_test, baseline.predict(X_test.toarray()))
            hitl_acc = accuracy_score(y_test, hitl.predict(X_test.toarray()))
            
            # Display metrics
            cols = st.columns(2)
            cols[0].metric("Baseline Accuracy", f"{base_acc:.1%}")
            cols[1].metric(
                "HITL Accuracy", 
                f"{hitl_acc:.1%}", 
                f"Δ{hitl_acc-base_acc:+.1%}"
            )
            
            # Confusion matrices
            st.subheader("Confusion Matrices")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            plot_confusion_matrix(baseline, X_test, y_test, ax=ax1, title="Baseline")
            plot_confusion_matrix(hitl, X_test, y_test, ax=ax2, title="HITL")
            st.pyplot(fig)
            
        except Exception as e:
            st.warning(f"Performance analysis unavailable: {str(e)}")

if __name__ == "__main__":
    main()