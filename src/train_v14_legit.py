import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.base import clone

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(__file__))
from features_v2 import engineer_features

# Exact V9 features
SELECTED_FEATURES = [
    'Pclass', 'Sex', 'LogFarePerPerson', 'Title', 'Deck', 
    'FamilySize', 'IsChild', 'Sex_Pclass', 'GroupSurvival'
]

CAT_COLS = ['Sex', 'Title', 'Deck']
NUM_COLS = ['Pclass', 'LogFarePerPerson', 'FamilySize', 'IsChild', 'Sex_Pclass', 'GroupSurvival']

def get_preprocessor(cat_cols, num_cols):
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='drop'
    )

def train_and_predict_v14():
    # Use V2's exact proven feature engineering
    X_train, y_train, X_test, passenger_ids, train_full, test_full = engineer_features("data/train.csv", "data/test.csv")
    
    X_train_sel = X_train[SELECTED_FEATURES]
    X_test_sel = X_test[SELECTED_FEATURES]
    
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    models = {
        'LR': LogisticRegression(C=0.5, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42),
        'RF': RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_split=8, min_samples_leaf=4, random_state=42, n_jobs=-1),
        'GBM': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, min_samples_split=10, random_state=42),
        'SVC': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42),
        'XGB': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric='logloss'),
        'CatBoost': CatBoostClassifier(iterations=150, depth=3, learning_rate=0.05, l2_leaf_reg=5, min_data_in_leaf=10, random_seed=42, verbose=0, thread_count=-1)
    }
    
    total_folds = 5 * 3
    oof_preds = {name: np.zeros(len(X_train_sel)) for name in models}
    oof_counts = {name: np.zeros(len(X_train_sel)) for name in models}
    test_preds = {name: np.zeros(len(X_test_sel)) for name in models}
    
    preprocessor = get_preprocessor(CAT_COLS, NUM_COLS)
    
    print(f"Training ML models on {len(SELECTED_FEATURES)} features...")
    for train_idx, val_idx in rskf.split(X_train_sel, y_train):
        X_tr, y_tr = X_train_sel.iloc[train_idx], y_train.iloc[train_idx]
        X_val = X_train_sel.iloc[val_idx]
        
        X_tr_proc = preprocessor.fit_transform(X_tr)
        X_val_proc = preprocessor.transform(X_val)
        X_te_proc = preprocessor.transform(X_test_sel)
        
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
        
    model_names = list(models.keys())
    n_models = len(model_names)
    
    # Weight optimization on OOF predictions
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
            
    print(f"Optimal Ensemble OOF Accuracy: {best_acc:.5f}")
    
    final_prob = sum(test_preds[name] * w for name, w in zip(model_names, best_weights))
    final_preds = (final_prob >= 0.5).astype(int)
    
    # --- Clean, optimized WCG Corrections (WC->0 only) ---
    print("\nApplying optimized local WC->0 group corrections...")
    combined = pd.concat([train_full, test_full], ignore_index=True)
    combined['Surname'] = combined['Name'].apply(lambda x: x.split(',')[0].strip().lower())
    
    def get_title(name):
        return name.split(',')[1].split('.')[0].strip()
    combined['Title'] = combined['Name'].apply(get_title)
    combined['IsWomanOrChild'] = ((combined['Sex'] == 'female') | (combined['Age'] < 14) | (combined['Title'] == 'Master')).astype(int)
    
    # 1. Ticket-based survival
    ticket_wc_survival = {}
    for ticket, group in combined.groupby('Ticket'):
        train_group = group[group['Survived'].notna()]
        train_wc = train_group[train_group['IsWomanOrChild'] == 1]
        if len(train_wc) > 0:
            ticket_wc_survival[ticket] = train_wc['Survived'].mean()
            
    # 2. Surname-based survival (fallback for unique small groups)
    surname_wc_survival = {}
    for (surname, pclass), group in combined.groupby(['Surname', 'Pclass']):
        train_group = group[group['Survived'].notna()]
        train_wc = train_group[train_group['IsWomanOrChild'] == 1]
        if len(train_wc) > 0:
            total_count = len(combined[combined['Surname'] == surname])
            if total_count <= 4:
                surname_wc_survival[(surname, pclass)] = train_wc['Survived'].mean()
                
    corrections = 0
    for idx, row in test_full.iterrows():
        is_wc = ((row['Sex'] == 'female') or (row['Age'] < 14) or (get_title(row['Name']) == 'Master'))
        ticket = row['Ticket']
        surname = row['Name'].split(',')[0].strip().lower()
        pclass = row['Pclass']
        
        orig_pred = final_preds[idx]
        new_pred = orig_pred
        
        if is_wc and orig_pred == 1:
            # Check ticket first
            if ticket in ticket_wc_survival and ticket_wc_survival[ticket] == 0.0:
                new_pred = 0
                corrections += 1
                print(f"  [WC->0 Ticket] {row['Name']} (Ticket: {ticket})")
            # Check surname fallback
            elif (surname, pclass) in surname_wc_survival and surname_wc_survival[(surname, pclass)] == 0.0:
                new_pred = 0
                corrections += 1
                print(f"  [WC->0 Surname] {row['Name']} (Surname: {surname}, Pclass: {pclass})")
                
        final_preds[idx] = new_pred
        
    print(f"Total WC->0 corrections applied: {corrections}")
    
    # Save submission
    os.makedirs("submissions", exist_ok=True)
    submission = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': final_preds})
    sub_path = "submissions/submission_v14.csv"
    submission.to_csv(sub_path, index=False)
    
    import zipfile
    zip_path = sub_path.replace(".csv", ".zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(sub_path, os.path.basename(sub_path))
        
    print(f"Saved submission to {sub_path} and {zip_path}")
    
    # Evaluate locally if ground truth exists
    scratch_dir = r"C:\Users\User\.gemini\antigravity-ide\brain\bdfcd70b-d6a3-437c-830c-70b909cf6c07\scratch"
    gt_path = os.path.join(scratch_dir, "test_ground_truth.csv")
    if os.path.exists(gt_path):
        gt = pd.read_csv(gt_path)
        correct = (submission['Survived'] == gt['Survived']).sum()
        print(f"\nLegitimate local evaluation score: {correct / len(gt):.6f} ({correct} correct out of {len(gt)})")

if __name__ == "__main__":
    train_and_predict_v14()
