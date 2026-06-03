"""
Titanic V13 - Multi-seed bootstrap ensemble of V9's EXACT proven architecture.
Strategy: Run V9's proven pipeline (same 6 models, same 9 features, same V2 features)
with 10 different random seeds and AVERAGE the test probabilities.

Why this works: Reduces prediction variance without introducing new bias.
V9 scored 0.79904 - the best so far. Everything that added complexity made it worse.
We need ~3 more correct predictions out of 418. Variance reduction via seed averaging
can shift borderline predictions more reliably.
"""
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
from train_v2 import apply_family_corrections

# ── EXACT V9 feature set ──────────────────────────────────────────────
SELECTED_FEATURES = [
    'Pclass', 'Sex', 'LogFarePerPerson', 'Title', 'Deck',
    'FamilySize', 'IsChild', 'Sex_Pclass', 'GroupSurvival'
]
CAT_COLS = ['Sex', 'Title', 'Deck']
NUM_COLS = ['Pclass', 'LogFarePerPerson', 'FamilySize',
            'IsChild', 'Sex_Pclass', 'GroupSurvival']

# 10 independent seeds — gives 10 × 6 models × 15 folds = 900 unique fits
SEEDS = [42, 7, 13, 99, 123, 256, 314, 512, 777, 1000]


def run_single_seed(X_train, y_train, X_test, seed):
    """Run the full V9 pipeline with a given random seed. Returns test probs."""

    def make_models(s):
        return {
            'LR': LogisticRegression(
                C=0.5, solver='lbfgs', max_iter=1000, random_state=s),
            'RF': RandomForestClassifier(
                n_estimators=300, max_depth=5, min_samples_split=8,
                min_samples_leaf=4, max_features='sqrt', random_state=s, n_jobs=-1),
            'GBM': GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8,
                min_samples_split=10, min_samples_leaf=5, random_state=s),
            'SVC': SVC(
                C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=s),
            'XGB': XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0,
                min_child_weight=5, random_state=s, n_jobs=-1,
                eval_metric='logloss', verbosity=0),
            'CatBoost': CatBoostClassifier(
                iterations=150, depth=3, learning_rate=0.05, l2_leaf_reg=5,
                min_data_in_leaf=10, random_seed=s, verbose=0, thread_count=-1),
        }

    models = make_models(seed)
    model_names = list(models.keys())
    n_models = len(model_names)
    total_folds = 15  # 5 splits × 3 repeats — same as V9

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS)
        ],
        remainder='drop'
    )

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)

    oof_preds  = {name: np.zeros(len(X_train)) for name in model_names}
    oof_counts = {name: np.zeros(len(X_train)) for name in model_names}
    test_preds = {name: np.zeros(len(X_test))  for name in model_names}

    for train_idx, val_idx in rskf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val      = X_train.iloc[val_idx]

        X_tr_p  = preprocessor.fit_transform(X_tr)
        X_val_p = preprocessor.transform(X_val)
        X_te_p  = preprocessor.transform(X_test)

        for name, clf in models.items():
            m = clone(clf)
            m.fit(X_tr_p, y_tr)
            oof_preds[name][val_idx]  += m.predict_proba(X_val_p)[:, 1]
            oof_counts[name][val_idx] += 1
            test_preds[name]          += m.predict_proba(X_te_p)[:, 1] / total_folds

    for name in model_names:
        mask = oof_counts[name] > 0
        oof_preds[name][mask] /= oof_counts[name][mask]

    # Weight optimisation — same 20k dirichlet search as V9
    best_acc, best_weights = 0.0, None
    np.random.seed(seed)
    for _ in range(20000):
        w = np.random.dirichlet(np.ones(n_models))
        ep = sum(oof_preds[n] * wi for n, wi in zip(model_names, w))
        acc = accuracy_score(y_train, (ep >= 0.5).astype(int))
        if acc > best_acc:
            best_acc, best_weights = acc, w

    test_prob = sum(test_preds[n] * wi for n, wi in zip(model_names, best_weights))
    return test_prob, best_acc


def train_and_predict_v13():
    print("Loading and engineering features (V2 pipeline)...")
    X_train, y_train, X_test, passenger_ids, train_full, test_full = engineer_features(
        "data/train.csv", "data/test.csv"
    )

    X_train = X_train[SELECTED_FEATURES]
    X_test  = X_test[SELECTED_FEATURES]

    print(f"Training with {len(SELECTED_FEATURES)} features, "
          f"{len(SEEDS)} seeds × 6 models × 15 folds = "
          f"{len(SEEDS)*6*15} total fits\n")

    all_probs = []
    seed_accs = []

    for i, seed in enumerate(SEEDS, 1):
        print(f"[{i:02d}/{len(SEEDS)}] seed={seed}", end="  ", flush=True)
        prob, acc = run_single_seed(X_train, y_train, X_test, seed)
        all_probs.append(prob)
        seed_accs.append(acc)
        print(f"OOF={acc:.5f}  survivors_predicted={int((prob>=0.5).sum())}")

    print(f"\nMean OOF across seeds: {np.mean(seed_accs):.5f}")
    print(f"Std  OOF across seeds: {np.std(seed_accs):.5f}")

    # Average probabilities across all seeds (key step)
    avg_prob    = np.mean(all_probs, axis=0)
    final_preds = (avg_prob >= 0.5).astype(int)

    print(f"\nPredicted survivors before corrections: {final_preds.sum()}")

    # Apply V2 family consistency corrections (same as V9)
    print("Applying V2 family corrections...")
    final_preds = apply_family_corrections(test_full, final_preds, passenger_ids.values)

    print(f"Predicted survivors after  corrections: {final_preds.sum()}")

    os.makedirs("submissions", exist_ok=True)
    submission = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': final_preds})
    sub_path   = "submissions/submission_v13.csv"
    submission.to_csv(sub_path, index=False)

    print(f"\nSaved: {sub_path}")
    print(f"Rows:  {len(submission)}")
    print(f"Distribution: {submission['Survived'].value_counts().to_dict()}")
    print(f"Survival rate: {final_preds.mean():.4f}")


if __name__ == "__main__":
    train_and_predict_v13()
