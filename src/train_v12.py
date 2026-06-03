"""
Titanic V12 - Clean rebuild from V9 best practices.
Key improvements over V9:
  1. Stacking meta-learner (LR trained on OOF) instead of random weight search
  2. ExtraTrees added to ensemble for diversity
  3. Ultra-conservative WCG post-processing: only corrects predictions for passengers
     with 100% clear group signal from training data AND borderline ML probability
  4. Fixed deprecation warnings (no penalty='l2')
  5. No combined-dataset Family_Survival leakage (was the V11 bug)
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Import the proven V2 feature engineering (clean, no leakage)
sys.path.insert(0, os.path.dirname(__file__))
from features_v2 import engineer_features


def get_preprocessor(cat_cols, num_cols):
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='drop'
    )


def apply_conservative_wcg(test_df, predictions, pred_probs, passenger_ids, train_df):
    """
    Ultra-conservative WCG post-processing.
    ONLY corrects predictions when ALL of:
    1. Test passenger shares ticket with training passengers
    2. ALL women+children in that training ticket group have the SAME survival outcome
    3. The current ML probability is borderline (between 0.30 and 0.70)
    
    This avoids the V10 mistake of overriding everyone.
    """
    pred_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions.copy(),
        'Prob': pred_probs
    })
    pred_df = pred_df.merge(
        test_df[['PassengerId', 'Ticket', 'Sex', 'Age', 'Title']],
        on='PassengerId', how='left'
    )

    corrections = 0

    # Build training ticket group signals
    ticket_signal = {}
    for ticket, grp in train_df.groupby('Ticket'):
        wc = grp[(grp['Sex'] == 'female') | (grp['Age'] < 14)]
        if len(wc) >= 1:
            if wc['Survived'].min() == wc['Survived'].max():  # unanimous
                ticket_signal[ticket] = int(wc['Survived'].iloc[0])

    for i, row in pred_df.iterrows():
        ticket = row['Ticket']
        if ticket not in ticket_signal:
            continue  # no clear signal, keep ML prediction
        signal = ticket_signal[ticket]
        prob = row['Prob']
        # Only intervene for borderline predictions
        if 0.30 <= prob <= 0.70:
            if signal == 1 and row['Survived'] == 0:
                pred_df.loc[i, 'Survived'] = 1
                corrections += 1
            elif signal == 0 and row['Survived'] == 1:
                pred_df.loc[i, 'Survived'] = 0
                corrections += 1

    print(f"  Conservative WCG corrections applied: {corrections}")
    return pred_df['Survived'].values


def train_and_predict_v12():
    # Use V2's exact proven feature engineering (same as V9)
    X_train, y_train, X_test, passenger_ids, train_full, test_full = engineer_features(
        "data/train.csv", "data/test.csv"
    )

    # Same strict feature selection as V9 (proven to prevent overfitting)
    selected_features = [
        'Pclass', 'Sex', 'LogFarePerPerson', 'Title', 'Deck',
        'FamilySize', 'IsChild', 'Sex_Pclass', 'GroupSurvival'
    ]

    cat_cols = ['Sex', 'Title', 'Deck']
    num_cols = ['Pclass', 'LogFarePerPerson', 'FamilySize', 'IsChild', 'Sex_Pclass', 'GroupSurvival']

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    total_folds = 5 * 5

    models = {
        'LR': LogisticRegression(C=0.5, solver='lbfgs', max_iter=1000, random_state=42),
        'RF': RandomForestClassifier(
            n_estimators=500, max_depth=5, min_samples_split=8,
            min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'ET': ExtraTreesClassifier(
            n_estimators=500, max_depth=6, min_samples_split=6,
            min_samples_leaf=3, max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'GBM': GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
            min_samples_split=10, min_samples_leaf=5, random_state=42
        ),
        'SVC': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42),
        'XGB': XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0,
            min_child_weight=5, random_state=42, n_jobs=-1, eval_metric='logloss',
            verbosity=0
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

    n_models = len(models)
    model_names = list(models.keys())

    oof_preds = {name: np.zeros(len(X_train)) for name in models}
    oof_counts = {name: np.zeros(len(X_train)) for name in models}
    test_preds = {name: np.zeros(len(X_test)) for name in models}

    # OOF matrix for stacking
    oof_matrix = np.zeros((len(X_train), n_models))
    test_matrix = np.zeros((len(X_test), n_models))

    preprocessor = get_preprocessor(cat_cols, num_cols)

    print(f"Training V12 with {len(selected_features)} features, {total_folds} folds, {n_models} models...")

    fold_num = 0
    for train_idx, val_idx in rskf.split(X_train, y_train):
        fold_num += 1
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]

        X_tr_proc = preprocessor.fit_transform(X_tr)
        X_val_proc = preprocessor.transform(X_val)
        X_te_proc = preprocessor.transform(X_test)

        for mi, (name, clf) in enumerate(models.items()):
            clf_fold = clone(clf)
            clf_fold.fit(X_tr_proc, y_tr)

            val_probs = clf_fold.predict_proba(X_val_proc)[:, 1]
            oof_preds[name][val_idx] += val_probs
            oof_counts[name][val_idx] += 1

            test_probs = clf_fold.predict_proba(X_te_proc)[:, 1]
            test_preds[name] += test_probs / total_folds

        if fold_num % 5 == 0:
            print(f"  Repeat {fold_num // 5}/{5} complete")

    # Average OOF
    for mi, name in enumerate(model_names):
        mask = oof_counts[name] > 0
        oof_preds[name][mask] /= oof_counts[name][mask]
        oof_matrix[:, mi] = oof_preds[name]
        test_matrix[:, mi] = test_preds[name]

    # Individual model scores
    print("\n--- Individual OOF Scores ---")
    for name in model_names:
        preds = (oof_preds[name] >= 0.5).astype(int)
        score = accuracy_score(y_train, preds)
        print(f"  {name}: {score:.5f}")

    # --- Method 1: Random weight search (V9 style) ---
    print("\n--- Method 1: Random Weight Search (30k iters) ---")
    best_acc_weights = 0.0
    best_weights = None
    np.random.seed(42)
    for _ in range(30000):
        weights = np.random.dirichlet(np.ones(n_models))
        ensemble_prob = sum(oof_preds[name] * w for name, w in zip(model_names, weights))
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)
        acc = accuracy_score(y_train, ensemble_pred)
        if acc > best_acc_weights:
            best_acc_weights = acc
            best_weights = weights
    print(f"  Best OOF (weights): {best_acc_weights:.5f}")

    # --- Method 2: Stacking meta-learner ---
    print("\n--- Method 2: Stacking Meta-Learner ---")
    meta_model = LogisticRegression(C=0.1, solver='lbfgs', max_iter=1000, random_state=42)
    meta_model.fit(oof_matrix, y_train)
    meta_oof_pred = meta_model.predict(oof_matrix)
    meta_oof_prob = meta_model.predict_proba(oof_matrix)[:, 1]
    meta_acc = accuracy_score(y_train, meta_oof_pred)
    print(f"  Best OOF (stacking): {meta_acc:.5f}")

    # Pick the better method
    if best_acc_weights >= meta_acc:
        print(f"\nUsing weighted ensemble (OOF: {best_acc_weights:.5f})")
        final_prob = sum(test_preds[name] * w for name, w in zip(model_names, best_weights))
        best_overall_acc = best_acc_weights
    else:
        print(f"\nUsing stacking meta-learner (OOF: {meta_acc:.5f})")
        final_prob = meta_model.predict_proba(test_matrix)[:, 1]
        best_overall_acc = meta_acc

    # Threshold optimization on OOF
    oof_ensemble_prob = sum(oof_preds[name] * w for name, w in zip(model_names, best_weights))
    best_threshold = 0.5
    best_threshold_acc = best_overall_acc
    for t in np.arange(0.38, 0.62, 0.005):
        pred = (oof_ensemble_prob >= t).astype(int)
        acc = accuracy_score(y_train, pred)
        if acc > best_threshold_acc:
            best_threshold_acc = acc
            best_threshold = t
    print(f"\nBest threshold: {best_threshold:.3f} (OOF: {best_threshold_acc:.5f})")

    final_preds = (final_prob >= best_threshold).astype(int)

    # --- Conservative WCG post-processing ---
    print("\n--- Applying Conservative WCG Post-Processing ---")
    train_raw = pd.read_csv("data/train.csv")
    train_raw['Title'] = train_raw['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    final_preds = apply_conservative_wcg(
        test_full, final_preds, final_prob, passenger_ids.values, train_raw
    )

    # Save
    os.makedirs("submissions", exist_ok=True)
    submission = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': final_preds})
    sub_path = "submissions/submission_v12.csv"
    submission.to_csv(sub_path, index=False)

    print(f"\nSaved to {sub_path}")
    print(f"Total rows: {len(submission)}")
    dist = submission['Survived'].value_counts(normalize=True).round(4)
    print(f"Class distribution:\n{dist}")
    print(f"Survival rate: {final_preds.mean():.4f}")


if __name__ == "__main__":
    train_and_predict_v12()
