import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.base import clone

from features_v3 import engineer_features

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def get_preprocessor(cat_cols, num_cols):
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='drop'
    )


def train_and_predict(X_train, y_train, X_test, passenger_ids, train_df_full, test_df_full,
                      cat_cols, num_cols, n_splits=5, n_repeats=3):
    """
    Trains multiple classifiers using Repeated Stratified K-Fold CV.
    Uses n_repeats to stabilize prediction variance.
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    
    # Highly regularized base models
    models = {
        'LR': LogisticRegression(C=0.8, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42),
        'RF': RandomForestClassifier(
            n_estimators=500, max_depth=5, min_samples_split=8,
            min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'GBM': GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
            min_samples_split=10, min_samples_leaf=5, random_state=42
        ),
        'SVC': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42),
        'XGB': XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0, 
            min_child_weight=5, random_state=42, n_jobs=-1, eval_metric='logloss'
        ),
        'LGBM': LGBMClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0,
            min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=200, depth=3, learning_rate=0.05, l2_leaf_reg=5,
            min_data_in_leaf=10, random_seed=42, verbose=0, thread_count=-1
        )
    }
    
    total_folds = n_splits * n_repeats
    oof_preds = {name: np.zeros(len(X_train)) for name in models}
    oof_counts = {name: np.zeros(len(X_train)) for name in models}
    test_preds = {name: np.zeros(len(X_test)) for name in models}
    
    preprocessor = get_preprocessor(cat_cols, num_cols)
    
    print(f"\n--- Running Stratified {n_splits}-Fold x {n_repeats}-Repeat CV ---")
    
    fold_num = 0
    for train_idx, val_idx in rskf.split(X_train, y_train):
        fold_num += 1
        
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        X_tr_proc = preprocessor.fit_transform(X_tr)
        X_val_proc = preprocessor.transform(X_val)
        X_te_proc = preprocessor.transform(X_test)
        
        for name, clf in models.items():
            clf_fold = clone(clf)
            clf_fold.fit(X_tr_proc, y_tr)
            
            val_probs = clf_fold.predict_proba(X_val_proc)[:, 1]
            oof_preds[name][val_idx] += val_probs
            oof_counts[name][val_idx] += 1
            
            test_probs = clf_fold.predict_proba(X_te_proc)[:, 1]
            test_preds[name] += test_probs / total_folds
        
        if fold_num % n_splits == 0:
            repeat = fold_num // n_splits
            print(f"  Repeat {repeat}/{n_repeats} complete")
            
    # Average OOF predictions
    for name in models:
        mask = oof_counts[name] > 0
        oof_preds[name][mask] /= oof_counts[name][mask]
        
    # Evaluate OOF accuracy of individual models
    print("\n=== Out-of-Fold Model Accuracies ===")
    for name in models:
        preds = (oof_preds[name] >= 0.5).astype(int)
        score = accuracy_score(y_train, preds)
        print(f"  {name}: {score:.5f}")
        
    # Optimize soft-voting ensemble weights
    print("\n--- Optimizing Soft-Voting Ensemble Weights ---")
    model_names = list(models.keys())
    n_models = len(model_names)
    
    best_acc = 0.0
    best_weights = None
    np.random.seed(42)
    
    for _ in range(30000):
        weights = np.random.dirichlet(np.ones(n_models))
        ensemble_prob = sum(oof_preds[name] * w for name, w in zip(model_names, weights))
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)
        acc = accuracy_score(y_train, ensemble_pred)
        if acc > best_acc:
            best_acc = acc
            best_weights = weights
            
    print(f"Best Ensemble ML-only OOF Accuracy: {best_acc:.5f}")
    print("Optimal Weights:")
    for name, w in zip(model_names, best_weights):
        print(f"  {name}: {w:.4f}")
        
    # Generate final ensemble predictions
    final_prob = sum(test_preds[name] * w for name, w in zip(model_names, best_weights))
    final_preds = (final_prob >= 0.5).astype(int)
    
    # Save ML-only submission file
    os.makedirs("submissions", exist_ok=True)
    submission = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': final_preds})
    ml_submission_path = "submissions/submission_ml_only_v3.csv"
    submission.to_csv(ml_submission_path, index=False)
    print(f"\nML-only submission saved to: {ml_submission_path}")
    
    # Save training metadata and models metadata
    os.makedirs("metadata", exist_ok=True)
    with open("metadata/oof_v3.pkl", "wb") as f:
        pickle.dump({'oof': oof_preds, 'test': test_preds, 'weights': best_weights}, f)
    print("Metadata saved to metadata/oof_v3.pkl")
    
    return ml_submission_path


if __name__ == "__main__":
    X_train, y_train, X_test, pids, train_full, test_full = engineer_features("data/train.csv", "data/test.csv")
    
    cat_cols = ['Sex', 'Embarked', 'Title', 'Deck', 'TicketPrefix']
    num_cols = ['Age', 'SibSp', 'Parch', 'LogFare', 'LogFarePerPerson',
                'FamilySize', 'TicketGroupSize', 'HasCabin', 'IsChild',
                'IsYoungAdult', 'IsAlone', 'SmallFamily', 'Sex_Pclass', 'Age_Pclass']
                
    train_and_predict(X_train, y_train, X_test, pids, train_full, test_full, cat_cols, num_cols)
