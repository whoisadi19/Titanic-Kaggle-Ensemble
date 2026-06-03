import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_data_and_predictions():
    """Loads target variable and OOF/Test predictions."""
    # Load actual targets from train.csv
    train = pd.read_csv("data/train.csv")
    y_train = train['Survived'].values
    
    # Load test passenger IDs
    test = pd.read_csv("data/test.csv")
    passenger_ids = test['PassengerId'].values
    
    # Load predictions
    with open("metadata/oof_predictions.pkl", "rb") as f:
        oof_preds = pickle.load(f)
    with open("metadata/test_predictions.pkl", "rb") as f:
        test_preds = pickle.load(f)
        
    return y_train, passenger_ids, oof_preds, test_preds

def analyze_correlations(oof_preds):
    """Prints the correlation matrix between model predictions to ensure diversity."""
    df_oof = pd.DataFrame(oof_preds)
    corr = df_oof.corr()
    print("\n--- Model Prediction Correlation Matrix ---")
    print(corr.round(4))
    print("-------------------------------------------")
    return corr

def find_optimal_weights(oof_preds, y_train):
    """
    Performs a randomized search to find the optimal soft-voting ensemble weights
    that maximize the cross-validated accuracy.
    """
    model_names = list(oof_preds.keys())
    n_models = len(model_names)
    
    best_acc = 0.0
    best_weights = None
    
    print("\nSearching for optimal ensemble weights...")
    
    # Run a random search over 10,000 weight combinations
    np.random.seed(42)
    for _ in range(10000):
        # Generate random weights summing to 1
        weights = np.random.dirichlet(np.ones(n_models))
        
        # Calculate weighted average OOF probability
        ensemble_prob = np.zeros(len(y_train))
        for i, name in enumerate(model_names):
            ensemble_prob += oof_preds[name] * weights[i]
            
        # Evaluate accuracy
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)
        acc = accuracy_score(y_train, ensemble_pred)
        
        if acc > best_acc:
            best_acc = acc
            best_weights = weights
            
    print(f"\nBest Ensemble OOF Accuracy: {best_acc:.5f}")
    print("Optimal Model Weights:")
    for name, weight in zip(model_names, best_weights):
        print(f" - {name}: {weight:.4f}")
        
    return best_weights

def generate_submission(test_preds, weights, passenger_ids):
    """Generates the final submission CSV file using optimal weights."""
    model_names = list(test_preds.keys())
    
    # Calculate weighted average test probability
    final_prob = np.zeros(len(passenger_ids))
    for i, name in enumerate(model_names):
        final_prob += test_preds[name] * weights[i]
        
    # Convert to binary outcomes
    final_preds = (final_prob >= 0.5).astype(int)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': final_preds
    })
    
    os.makedirs("submissions", exist_ok=True)
    submission_path = "submissions/submission_ensemble.csv"
    submission.to_csv(submission_path, index=False)
    
    print(f"\nSuccessfully generated submission file: {submission_path}")
    print(f"Total rows: {len(submission)}")
    print(f"Class distribution:\n{submission['Survived'].value_counts(normalize=True).round(4)}")
    
    # Quick sanity check on the first few rows
    print("\nFirst 10 rows of submission:")
    print(submission.head(10))
    
    return submission_path

if __name__ == "__main__":
    # Check if files exist
    if not (os.path.exists("metadata/oof_predictions.pkl") and os.path.exists("metadata/test_predictions.pkl")):
        print("Error: OOF predictions not found. Run train.py first.")
        exit(1)
        
    y_train, passenger_ids, oof_preds, test_preds = load_data_and_predictions()
    
    # Analyze correlation of base models
    analyze_correlations(oof_preds)
    
    # Optimize soft-voting weights
    best_weights = find_optimal_weights(oof_preds, y_train)
    
    # Generate final ensemble predictions
    generate_submission(test_preds, best_weights, passenger_ids)
