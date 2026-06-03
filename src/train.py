import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Import feature engineering function
from features import engineer_features

# Import ML models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def get_preprocessor(cat_cols, num_cols):
    """Creates a scikit-learn ColumnTransformer for preprocessing."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='passthrough'
    )
    return preprocessor

def train_base_models(X_train, y_train, X_test, cat_cols, num_cols, n_splits=10):
    """
    Trains multiple ML classifiers using Stratified K-Fold cross validation.
    Tracks Out-of-Fold (OOF) predictions and aggregates test-set predictions.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Define models with hyperparameters tuned for small datasets (regularized to prevent overfitting)
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_split=5, 
            min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'SVC': SVC(
            C=1.5, kernel='rbf', gamma='scale', probability=True, random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0, random_state=42,
            n_jobs=-1, eval_metric='logloss'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0, random_state=42,
            n_jobs=-1, verbose=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=300, depth=4, learning_rate=0.03, l2_leaf_reg=3,
            random_seed=42, verbose=0, thread_count=-1
        )
    }
    
    # Initialize dictionary to store out-of-fold predictions
    oof_preds = {name: np.zeros(len(X_train)) for name in models.keys()}
    
    # Initialize dictionary to store aggregated test predictions
    test_preds = {name: np.zeros(len(X_test)) for name in models.keys()}
    
    # Preprocessing
    preprocessor = get_preprocessor(cat_cols, num_cols)
    
    print(f"\n--- Starting {n_splits}-Fold Stratified Cross-Validation ---")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        
        # Split features and labels
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # Fit preprocessor on training fold and transform both train and val folds
        X_tr_proc = preprocessor.fit_transform(X_tr)
        X_val_proc = preprocessor.transform(X_val)
        X_te_proc = preprocessor.transform(X_test)
        
        # Train each model
        for name, clf in models.items():
            # Clone model to avoid bleeding state
            from sklearn.base import clone
            clf_fold = clone(clf)
            
            clf_fold.fit(X_tr_proc, y_tr)
            
            # Predict validation probabilities
            val_probs = clf_fold.predict_proba(X_val_proc)[:, 1]
            oof_preds[name][val_idx] = val_probs
            
            # Predict test probabilities (accumulate for fold averaging)
            test_probs = clf_fold.predict_proba(X_te_proc)[:, 1]
            test_preds[name] += test_probs / n_splits
            
            # Print intermediate accuracy
            val_preds = (val_probs >= 0.5).astype(int)
            acc = accuracy_score(y_val, val_preds)
            print(f"Fold {fold + 1} - {name} Accuracy: {acc:.4f}")
            
    # Evaluation of overall OOF scores
    print("\n==========================================")
    print("      Overall Out-of-Fold Performances     ")
    print("==========================================")
    
    best_name = None
    best_score = 0.0
    
    for name in models.keys():
        preds = (oof_preds[name] >= 0.5).astype(int)
        score = accuracy_score(y_train, preds)
        print(f"\nModel: {name}")
        print(f"OOF Accuracy: {score:.5f}")
        print(classification_report(y_train, preds, digits=4))
        
        if score > best_score:
            best_score = score
            best_name = name
            
    # Create directories for models and metadata if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("metadata", exist_ok=True)
    
    # Save OOF and test predictions for ensembling
    with open("metadata/oof_predictions.pkl", "wb") as f:
        pickle.dump(oof_preds, f)
    with open("metadata/test_predictions.pkl", "wb") as f:
        pickle.dump(test_preds, f)
        
    print(f"\nSaved OOF and Test predictions to 'metadata/'.")
    print(f"Best single model is {best_name} with OOF Accuracy of {best_score:.5f}")
    
    return oof_preds, test_preds

if __name__ == "__main__":
    # Load and process data
    X_train, y_train, X_test, passenger_ids = engineer_features("data/train.csv", "data/test.csv")
    
    # Categorical and numerical columns
    cat_cols = ['Sex', 'Embarked', 'Title', 'Deck']
    num_cols = ['Age', 'SibSp', 'Parch', 'LogFare', 'LogTicketFare', 'FamilySize', 'MaxGroupSize']
    
    # PassengerId is ignored in modeling
    # Target is Survived
    
    # Run modeling pipeline
    oof_preds, test_preds = train_base_models(X_train, y_train, X_test, cat_cols, num_cols)
