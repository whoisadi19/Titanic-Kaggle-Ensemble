"""
Titanic V11 - Best of V9 ensemble + V10's Family_Survival as a MODEL FEATURE (not override)
Key changes from V10 (0.73923) and V9 (0.79904):
  - Family_Survival from V10 is used as an INPUT FEATURE, not a post-processing override
  - Uses the proven V2/V9 ensemble pipeline with 7 models
  - Adds Family_Survival_Ticket as a second group feature
  - Careful threshold tuning via OOF
  - NO aggressive rule-based overrides
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.base import clone

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def build_family_survival_deotte(train_df, test_df):
    """
    Build Chris Deotte's Family_Survival feature using combined train+test.
    Groups by Surname+Fare and Ticket to find family/group links.
    Returns the feature for both train and test.
    """
    full = pd.concat([train_df, test_df], ignore_index=True)
    full['Surname'] = full['Name'].apply(lambda x: x.split(',')[0].strip())
    
    full['Family_Survival'] = 0.5
    full['Family_Survival_Ticket'] = 0.5
    
    # Group by Surname + Fare
    for _, grp in full.groupby(['Surname', 'Fare']):
        if len(grp) > 1:
            for idx, row in grp.iterrows():
                smax = grp.drop(idx)['Survived'].max()
                smin = grp.drop(idx)['Survived'].min()
                if smax == 1.0:
                    full.loc[idx, 'Family_Survival'] = 1
                elif smin == 0.0:
                    full.loc[idx, 'Family_Survival'] = 0
    
    # Group by Ticket
    for _, grp in full.groupby('Ticket'):
        if len(grp) > 1:
            for idx, row in grp.iterrows():
                smax = grp.drop(idx)['Survived'].max()
                smin = grp.drop(idx)['Survived'].min()
                if smax == 1.0:
                    full.loc[idx, 'Family_Survival_Ticket'] = 1
                elif smin == 0.0:
                    full.loc[idx, 'Family_Survival_Ticket'] = 0
    
    # Combined: take the max signal (1 > 0.5 > 0)
    full['Family_Survival_Combined'] = np.where(
        full['Family_Survival'] != 0.5, full['Family_Survival'],
        full['Family_Survival_Ticket']
    )
    
    train_size = len(train_df)
    return (
        full.iloc[:train_size][['Family_Survival', 'Family_Survival_Ticket', 'Family_Survival_Combined']].values,
        full.iloc[train_size:][['Family_Survival', 'Family_Survival_Ticket', 'Family_Survival_Combined']].values
    )


def extract_title(name):
    title = name.split(',')[1].split('.')[0].strip()
    title_map = {
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss',
        'Lady': 'Royalty', 'the Countess': 'Royalty', 'Countess': 'Royalty',
        'Don': 'Royalty', 'Dona': 'Royalty', 'Sir': 'Royalty', 'Jonkheer': 'Royalty',
        'Capt': 'Officer', 'Col': 'Officer', 'Major': 'Officer',
        'Dr': 'Officer', 'Rev': 'Officer'
    }
    return title_map.get(title, title)


def engineer_v11(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    target = train['Survived'].copy()
    passenger_ids = test['PassengerId'].copy()
    
    # Build Family_Survival BEFORE modifying the dataframes
    train_fs, test_fs = build_family_survival_deotte(train, test)
    
    for df in [train, test]:
        # Title
        df['Title'] = df['Name'].apply(extract_title)
        
        # Age imputation by Title median
        title_age_map = train.groupby('Title')['Age'].median().to_dict()
        df['Age'] = df.apply(
            lambda r: title_age_map.get(r['Title'], train['Age'].median()) if pd.isna(r['Age']) else r['Age'],
            axis=1
        )
        
        # Fare imputation
        if df['Fare'].isna().any():
            fare_median = train.groupby('Pclass')['Fare'].median()
            df['Fare'] = df.apply(
                lambda r: fare_median.get(r['Pclass'], train['Fare'].median()) if pd.isna(r['Fare']) else r['Fare'],
                axis=1
            )
        
        # Embarked
        df['Embarked'] = df['Embarked'].fillna('S')
        
        # Cabin / Deck
        df['HasCabin'] = df['Cabin'].notna().astype(int)
        df['Deck'] = df['Cabin'].fillna('U').apply(lambda x: x[0]).replace('T', 'U')
        
        # Family size
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['SmallFamily'] = ((df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)).astype(int)
        
        # Age features
        df['IsChild'] = (df['Age'] < 14).astype(int)
        
        # Fare features
        df['LogFare'] = np.log1p(df['Fare'])
        
        # Interaction features (the most important one)
        df['Sex_Pclass'] = df['Sex'].map({'male': 0, 'female': 1}) * 3 + df['Pclass']
        df['Age_Pclass'] = df['Age'] * df['Pclass']
    
    # Ticket group size (from combined)
    combined = pd.concat([train, test], ignore_index=True)
    ticket_counts = combined['Ticket'].value_counts().to_dict()
    train['TicketGroupSize'] = train['Ticket'].map(ticket_counts)
    test['TicketGroupSize'] = test['Ticket'].map(ticket_counts)
    
    train['LogFarePerPerson'] = np.log1p(train['Fare'] / train['TicketGroupSize'])
    test['LogFarePerPerson'] = np.log1p(test['Fare'] / test['TicketGroupSize'])
    
    # Assign Family_Survival features
    train['Family_Survival'] = train_fs[:, 0]
    train['Family_Survival_Ticket'] = train_fs[:, 1]
    train['Family_Survival_Combined'] = train_fs[:, 2]
    
    test['Family_Survival'] = test_fs[:, 0]
    test['Family_Survival_Ticket'] = test_fs[:, 1]
    test['Family_Survival_Combined'] = test_fs[:, 2]
    
    # Ticket prefix
    def extract_ticket_prefix(ticket):
        parts = ticket.split()
        if len(parts) > 1:
            return parts[0].replace('.', '').replace('/', '').upper()
        return 'NONE'
    
    train['TicketPrefix'] = train['Ticket'].apply(extract_ticket_prefix)
    test['TicketPrefix'] = test['Ticket'].apply(extract_ticket_prefix)
    prefix_counts = train['TicketPrefix'].value_counts()
    common = prefix_counts[prefix_counts >= 5].index.tolist()
    train['TicketPrefix'] = train['TicketPrefix'].apply(lambda x: x if x in common else 'RARE')
    test['TicketPrefix'] = test['TicketPrefix'].apply(lambda x: x if x in common else 'RARE')
    
    features = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
        'LogFare', 'LogFarePerPerson',
        'Embarked', 'Title', 'Deck', 'TicketPrefix',
        'FamilySize', 'TicketGroupSize',
        'HasCabin', 'IsChild', 'IsAlone', 'SmallFamily',
        'Sex_Pclass', 'Age_Pclass',
        'Family_Survival', 'Family_Survival_Ticket', 'Family_Survival_Combined'
    ]
    
    return train[features], target, test[features], passenger_ids


def train_and_predict_v11():
    X_train, y_train, X_test, passenger_ids = engineer_v11("data/train.csv", "data/test.csv")
    
    cat_cols = ['Sex', 'Embarked', 'Title', 'Deck', 'TicketPrefix']
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='drop'
    )
    
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    
    models = {
        'LR': LogisticRegression(C=0.5, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42),
        'RF': RandomForestClassifier(
            n_estimators=500, max_depth=5, min_samples_split=8,
            min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'GBM': GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            min_samples_split=10, min_samples_leaf=5, random_state=42
        ),
        'SVC': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42),
        'XGB': XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0,
            min_child_weight=5, random_state=42, n_jobs=-1, eval_metric='logloss'
        ),
        'LGBM': LGBMClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0,
            min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=5,
            min_data_in_leaf=10, random_seed=42, verbose=0, thread_count=-1
        )
    }
    
    total_folds = 5 * 5
    oof_preds = {name: np.zeros(len(X_train)) for name in models}
    oof_counts = {name: np.zeros(len(X_train)) for name in models}
    test_preds = {name: np.zeros(len(X_test)) for name in models}
    
    print(f"Training V11 with {len(X_train.columns)} features, {total_folds} folds...")
    print(f"Features: {list(X_train.columns)}")
    
    fold_num = 0
    for train_idx, val_idx in rskf.split(X_train, y_train):
        fold_num += 1
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
        
        if fold_num % 5 == 0:
            print(f"  Repeat {fold_num // 5}/{5} complete")
    
    # Average OOF
    for name in models:
        mask = oof_counts[name] > 0
        oof_preds[name][mask] /= oof_counts[name][mask]
    
    # Print individual model scores
    print("\n--- Individual OOF Scores ---")
    for name in models:
        preds = (oof_preds[name] >= 0.5).astype(int)
        score = accuracy_score(y_train, preds)
        print(f"  {name}: {score:.5f}")
    
    # Optimize ensemble weights
    print("\n--- Optimizing Ensemble Weights (50k iterations) ---")
    model_names = list(models.keys())
    n_models = len(model_names)
    
    best_acc = 0.0
    best_weights = None
    np.random.seed(42)
    
    for _ in range(50000):
        weights = np.random.dirichlet(np.ones(n_models))
        ensemble_prob = sum(oof_preds[name] * w for name, w in zip(model_names, weights))
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)
        acc = accuracy_score(y_train, ensemble_pred)
        if acc > best_acc:
            best_acc = acc
            best_weights = weights
    
    print(f"  Best OOF Accuracy: {best_acc:.5f}")
    for name, w in zip(model_names, best_weights):
        print(f"    {name}: {w:.4f}")
    
    # Optimize threshold
    print("\n--- Optimizing Decision Threshold ---")
    oof_ensemble = sum(oof_preds[name] * w for name, w in zip(model_names, best_weights))
    best_threshold = 0.5
    best_threshold_acc = best_acc
    
    for t in np.arange(0.40, 0.60, 0.005):
        pred = (oof_ensemble >= t).astype(int)
        acc = accuracy_score(y_train, pred)
        if acc > best_threshold_acc:
            best_threshold_acc = acc
            best_threshold = t
    
    print(f"  Best Threshold: {best_threshold:.3f} (OOF Acc: {best_threshold_acc:.5f})")
    
    # Final predictions
    final_prob = sum(test_preds[name] * w for name, w in zip(model_names, best_weights))
    final_preds = (final_prob >= best_threshold).astype(int)
    
    # Save
    os.makedirs("submissions", exist_ok=True)
    submission = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': final_preds})
    sub_path = "submissions/submission_v11.csv"
    submission.to_csv(sub_path, index=False)
    
    print(f"\nSaved to {sub_path}")
    print(f"Total rows: {len(submission)}")
    print(f"Class distribution:\n{submission['Survived'].value_counts(normalize=True).round(4)}")
    print(f"\nSurvival rate: {final_preds.mean():.4f}")


if __name__ == "__main__":
    train_and_predict_v11()
