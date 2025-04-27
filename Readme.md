# Human-in-the-Loop (HITL) Spam Classifier 🚀📧

A machine learning system that combines AI predictions with human feedback to improve spam detection accuracy over time.

![Demo Screenshot](https://via.placeholder.com/800x400?text=HITL+Spam+Classifier+Demo) 
*(Replace with actual screenshot after deployment)*

## Features ✨

- **Active Learning**: Model requests human input for uncertain predictions
- **Real-time Feedback**: Immediate model updates from human corrections
- **Performance Tracking**: Compare baseline vs HITL-enhanced model
- **Intuitive Interface**: Streamlit web app for easy interaction
- **Persistent Learning**: Saves feedback and model improvements between sessions

## Requirements 📋

- Python 3.8+
- pip package manager

## Installation ⚙️

1. Clone the repository:
   ```bash
   git clone https://github.com/Ofgeha-Gelana/hitl-spam.git
   cd hitl-spam

   pip install -r requirements.txt

   python train_baseline.py

   streamlit run app.py