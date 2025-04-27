from utils.data_loader import load_data
from utils.model_utils import create_baseline_model, save_model

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, vectorizer, original_texts = load_data()
    
    # Train baseline model
    print("Training baseline model...")
    model = create_baseline_model(X_train, y_train)
    
    # Save model
    save_model(model, 'models/baseline_model.pkl')
    print("Baseline model trained and saved!")

if __name__ == "__main__":
    main()