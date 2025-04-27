import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from modAL.uncertainty import uncertainty_sampling
from utils.data_loader import load_data
from utils.model_utils import load_model
from active_learning import initialize_active_learner, save_feedback, save_active_learner

# Create required directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)

def init_session():
    """Initialize session state with comprehensive error handling"""
    try:
        if 'initialized' not in st.session_state:
            # Load data
            (X_train_vec, X_test_vec,
             y_train, y_test,
             vectorizer, original_texts) = load_data()
            
            # Store in session state
            st.session_state.update({
                'X_pool': X_train_vec,
                'y_pool': y_train,
                'vectorizer': vectorizer,
                'original_texts': original_texts,
                'initialized': True
            })
            
            # Initialize learner
            try:
                st.session_state.learner, st.session_state.feedback_df = initialize_active_learner()
            except Exception as e:
                st.error(f"Failed to initialize learner: {str(e)}")
                st.session_state.learner = None
                
    except Exception as e:
        st.error(f"Fatal error during initialization: {str(e)}")
        st.stop()

def get_new_sample():
    """Get the most uncertain sample from the pool with error handling"""
    try:
        if not hasattr(st.session_state, 'learner') or st.session_state.learner is None:
            raise ValueError("Learner not initialized")
            
        query_idx, query_inst = uncertainty_sampling(
            st.session_state.learner,
            st.session_state.X_pool
        )
        
        # Safely get text from Series
        text = str(st.session_state.original_texts.iloc[query_idx])
        if not text.strip():
            raise ValueError("Empty text sample retrieved")
            
        return text
        
    except Exception as e:
        st.error(f"Error getting sample: {str(e)}")
        return None

def plot_confusion_matrix(model, X, y, ax, title):
    """Plot confusion matrix with validation"""
    try:
        preds = model.predict(X.toarray())
        cm = confusion_matrix(y, preds)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    except Exception as e:
        print(f"Error plotting confusion matrix: {str(e)}")
        ax.set_title(f"Error: {str(e)}")

def main():
    st.set_page_config(
        page_title="HITL Spam Classifier",
        page_icon="📧",
        layout="wide"
    )
    
    # Initialize session
    init_session()
    
    # Check if initialization succeeded
    if not st.session_state.get('initialized', False) or st.session_state.learner is None:
        st.error("""
            Application failed to initialize. Possible causes:
            - Missing or invalid data file (data/spam.csv)
            - Corrupted model files
            - Insufficient system resources
        """)
        st.stop()
    
    st.title("📧 Human-in-the-Loop Spam Classifier")
    st.markdown("""
        Help improve our spam detection model by reviewing predictions!
        The system will learn from your feedback to become more accurate over time.
    """)
    
    # Sidebar stats
    with st.sidebar:
        st.header("📊 Dashboard")
        if not st.session_state.feedback_df.empty:
            total = len(st.session_state.feedback_df)
            corrections = sum(
                st.session_state.feedback_df['model_pred'] != 
                st.session_state.feedback_df['human_label']
            )
            st.metric("Total Reviews", total)
            st.metric("Corrections Made", corrections)
            st.metric("Correction Rate", f"{corrections/total:.1%}")
    
    # Main interface tabs
    tab1, tab2 = st.tabs(["💬 Provide Feedback", "📈 Performance Analysis"])
    
    with tab1:
        st.header("Provide Feedback on Predictions")
        
        if st.button("🎯 Get New Message to Review", type="primary"):
            sample_text = get_new_sample()
            if sample_text:
                st.session_state.current_text = sample_text
            else:
                st.warning("Could not retrieve a message to review")
        
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
                    
                    # Feedback buttons
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
                        if st.button("⏭ Skip", help="Skip this message if unsure"):
                            st.info("Message skipped")
                            st.session_state.current_text = None
                    
                    if st.session_state.get('show_correction', False):
                        correction = st.radio(
                            "What is the correct label?",
                            ['HAM (not spam)', 'SPAM'],
                            horizontal=True
                        )
                        
                        if st.button("Submit Correction"):
                            corrected_label = 1 if 'SPAM' in correction else 0
                            
                            # Teach the learner
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
                    st.error(f"Error processing message: {str(e)}")
    
    with tab2:
        st.header("Model Performance Analysis")
        
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
            plot_confusion_matrix(baseline, X_test, y_test, ax=ax1, title="Baseline Model")
            plot_confusion_matrix(hitl, X_test, y_test, ax=ax2, title="HITL Model")
            st.pyplot(fig)
            
        except Exception as e:
            st.warning(f"Couldn't generate performance analysis: {str(e)}")

if __name__ == "__main__":
    main()