import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.base import clone

from features_v3 import engineer_features_v3

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

def train_and_predict_v7():
    X_train, y_train, X_test, passenger_ids, train_full, test_full = engineer_features_v3("data/train.csv", "data/test.csv")
    
    cat_cols = ['Sex', 'Embarked', 'Title', 'Deck', 'TicketPrefix']
    num_cols = ['Age', 'SibSp', 'Parch', 'LogFare', 'LogFarePerPerson',
                'FamilySize', 'TicketGroupSize', 'GroupSurvival',
                'HasCabin', 'IsChild', 'IsYoungAdult', 'IsAlone', 'SmallFamily',
                'Sex_Pclass', 'Age_Pclass']
    
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    models = {
        'LR': LogisticRegression(C=0.8, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42),
        'RF': RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_split=8, min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1),
        'GBM': GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, min_samples_split=10, min_samples_leaf=5, random_state=42),
        'SVC': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42),
        'XGB': XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5, random_state=42, n_jobs=-1, eval_metric='logloss'),
        'LGBM': LGBMClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0, min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1),
        'CatBoost': CatBoostClassifier(iterations=200, depth=3, learning_rate=0.05, l2_leaf_reg=5, min_data_in_leaf=10, random_seed=42, verbose=0, thread_count=-1)
    }
    
    total_folds = 5 * 3
    oof_preds = {name: np.zeros(len(X_train)) for name in models}
    oof_counts = {name: np.zeros(len(X_train)) for name in models}
    test_preds = {name: np.zeros(len(X_test)) for name in models}
    
    preprocessor = get_preprocessor(cat_cols, num_cols)
    
    print("Training 5-Fold x 3-Repeat CV...")
    for train_idx, val_idx in rskf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        
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
            
    for name in models:
        mask = oof_counts[name] > 0
        oof_preds[name][mask] /= oof_counts[name][mask]
        
    print("Optimizing Weights...")
    model_names = list(models.keys())
    n_models = len(model_names)
    
    best_acc = 0.0
    best_weights = None
    np.random.seed(42)
    
    for _ in range(20000):
        weights = np.random.dirichlet(np.ones(n_models))
        ensemble_prob = sum(oof_preds[name] * w for name, w in zip(model_names, weights))
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)
        acc = accuracy_score(y_train, ensemble_pred)
        if acc > best_acc:
            best_acc = acc
            best_weights = weights
            
    print(f"Best Ensemble OOF Accuracy: {best_acc:.5f}")
    
    final_prob = sum(test_preds[name] * w for name, w in zip(model_names, best_weights))
    final_preds = (final_prob >= 0.5).astype(int)
    
    # Apply V2 Family Consistency Corrections
    from train_v2 import apply_family_corrections
    final_preds = apply_family_corrections(test_full, final_preds, passenger_ids.values)
    
    os.makedirs("submissions", exist_ok=True)
    submission = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': final_preds})
    sub_path = "submissions/submission_v7.csv"
    submission.to_csv(sub_path, index=False)
    
    import zipfile
    zip_path = sub_path.replace(".csv", ".zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(sub_path, os.path.basename(sub_path))
        
    print(f"Saved to {sub_path}")

if __name__ == "__main__":
    train_and_predict_v7()
