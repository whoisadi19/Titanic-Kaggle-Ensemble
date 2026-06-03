import pandas as pd
import numpy as np
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

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Extract Surname
train['Surname'] = train['Name'].apply(lambda x: x.split(',')[0].strip().lower())
test['Surname'] = test['Name'].apply(lambda x: x.split(',')[0].strip().lower())

# Extract Title
def extract_title(df):
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    title_mapping = {
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss',
        'Lady': 'Royalty', 'the Countess': 'Royalty', 'Countess': 'Royalty',
        'Don': 'Royalty', 'Dona': 'Royalty', 'Sir': 'Royalty', 'Jonkheer': 'Royalty',
        'Capt': 'Officer', 'Col': 'Officer', 'Major': 'Officer',
        'Dr': 'Officer', 'Rev': 'Officer'
    }
    df['Title'] = df['Title'].replace(title_mapping)
    return df

train = extract_title(train)
test = extract_title(test)

# Impute Age and Fare
def impute_age(df, median_ages=None):
    if median_ages is None:
        median_ages = df.groupby(['Pclass', 'Sex', 'Title'])['Age'].median().to_dict()
        fallback_1 = df.groupby(['Pclass', 'Sex'])['Age'].median().to_dict()
        fallback_overall = df['Age'].median()
    else:
        fallback_1 = median_ages.get('_fallback_1', {})
        fallback_overall = median_ages.get('_fallback_overall', 28.0)

    def get_age(row):
        if pd.notna(row['Age']):
            return row['Age']
        key = (row['Pclass'], row['Sex'], row['Title'])
        if key in median_ages and pd.notna(median_ages[key]):
            return median_ages[key]
        key2 = (row['Pclass'], row['Sex'])
        if key2 in fallback_1 and pd.notna(fallback_1[key2]):
            return fallback_1[key2]
        return fallback_overall

    df['Age'] = df.apply(get_age, axis=1)
    if '_fallback_1' not in median_ages:
        median_ages['_fallback_1'] = fallback_1
        median_ages['_fallback_overall'] = fallback_overall
    return df, median_ages

train, median_ages = impute_age(train)
test, _ = impute_age(test, median_ages)

fare_median = train.groupby('Pclass')['Fare'].median().to_dict()
train['Fare'] = train.apply(lambda r: fare_median.get(r['Pclass'], train['Fare'].median()) if pd.isna(r['Fare']) else r['Fare'], axis=1)
test['Fare'] = test.apply(lambda r: fare_median.get(r['Pclass'], train['Fare'].median()) if pd.isna(r['Fare']) else r['Fare'], axis=1)

# Categoricals and binary features
train['HasCabin'] = train['Cabin'].notna().astype(int)
test['HasCabin'] = test['Cabin'].notna().astype(int)
train['Deck'] = train['Cabin'].fillna('U').apply(lambda x: x[0]).replace('T', 'U')
test['Deck'] = test['Cabin'].fillna('U').apply(lambda x: x[0]).replace('T', 'U')

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
test['Embarked'] = test['Embarked'].fillna(train['Embarked'].mode()[0])

# Family sizes
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
train['IsAlone'] = (train['FamilySize'] == 1).astype(int)
test['IsAlone'] = (test['FamilySize'] == 1).astype(int)
train['SmallFamily'] = ((train['FamilySize'] >= 2) & (train['FamilySize'] <= 4)).astype(int)
test['SmallFamily'] = ((test['FamilySize'] >= 2) & (test['FamilySize'] <= 4)).astype(int)

# Sex_Pclass and Age_Pclass interactions
train['Sex_Pclass'] = train['Sex'].map({'male': 0, 'female': 1}) * 3 + train['Pclass']
test['Sex_Pclass'] = test['Sex'].map({'male': 0, 'female': 1}) * 3 + test['Pclass']
train['Age_Pclass'] = train['Age'] * train['Pclass']
test['Age_Pclass'] = test['Age'] * test['Pclass']
train['LogFare'] = np.log1p(train['Fare'])
test['LogFare'] = np.log1p(test['Fare'])

# Define group by Ticket or Surname
combined = pd.concat([train, test], ignore_index=True)
ticket_counts = combined['Ticket'].value_counts()

def get_group_id(row):
    ticket = row['Ticket']
    if ticket_counts[ticket] > 1:
        return f"ticket_{ticket}"
    
    surname = row['Surname']
    pclass = row['Pclass']
    embarked = row['Embarked']
    return f"family_{surname}_{pclass}_{embarked}"

combined['GroupId'] = combined.apply(get_group_id, axis=1)
combined['IsWomanOrChild'] = ((combined['Sex'] == 'female') | (combined['Name'].str.contains('Master.')) | (combined['Age'] < 14)).astype(int)

# Split back
train_df = combined[combined['Survived'].notna()].copy()
test_df = combined[combined['Survived'].isna()].copy()

# Base model and preprocessing setup
cat_cols = ['Sex', 'Embarked', 'Title', 'Deck']
num_cols = ['Age', 'SibSp', 'Parch', 'LogFare', 'FamilySize', 'HasCabin', 'IsAlone', 'SmallFamily', 'Sex_Pclass', 'Age_Pclass']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ],
    remainder='drop'
)

# Repeated Stratified K-Fold CV
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# Models to ensemble
models = {
    'LR': LogisticRegression(C=0.8, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42),
    'RF': RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_split=8, min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1),
    'GBM': GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, min_samples_split=10, min_samples_leaf=5, random_state=42),
    'SVC': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42),
    'XGB': XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5, random_state=42, n_jobs=-1, eval_metric='logloss'),
    'LGBM': LGBMClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0, min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1),
    'CatBoost': CatBoostClassifier(iterations=200, depth=3, learning_rate=0.05, l2_leaf_reg=5, min_data_in_leaf=10, random_seed=42, verbose=0, thread_count=-1)
}

model_names = list(models.keys())
n_models = len(model_names)

oof_probs = {name: np.zeros(len(train_df)) for name in models}
oof_counts = {name: np.zeros(len(train_df)) for name in models}

print("Running Repeated Stratified K-Fold CV...")
for train_idx, val_idx in rskf.split(train_df[num_cols + cat_cols], train_df['Survived']):
    X_tr = train_df.iloc[train_idx]
    y_tr = train_df.iloc[train_idx]['Survived']
    X_val = train_df.iloc[val_idx]
    
    X_tr_proc = preprocessor.fit_transform(X_tr)
    X_val_proc = preprocessor.transform(X_val)
    
    for name, clf in models.items():
        clf_fold = clone(clf)
        clf_fold.fit(X_tr_proc, y_tr)
        
        oof_probs[name][val_idx] += clf_fold.predict_proba(X_val_proc)[:, 1]
        oof_counts[name][val_idx] += 1

# Average out predictions
for name in models:
    oof_probs[name] /= oof_counts[name]

# Find optimal weights for soft-voting ensemble
np.random.seed(42)
best_acc = 0.0
best_weights = None
y_train_arr = train_df['Survived'].values

for _ in range(20000):
    weights = np.random.dirichlet(np.ones(n_models))
    ensemble_prob = sum(oof_probs[name] * w for name, w in zip(model_names, weights))
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)
    acc = accuracy_score(y_train_arr, ensemble_pred)
    if acc > best_acc:
        best_acc = acc
        best_weights = weights

# Final ensemble probability
final_ensemble_prob = sum(oof_probs[name] * w for name, w in zip(model_names, best_weights))
train_df['ML_Prob'] = final_ensemble_prob
train_df['ML_Pred'] = (train_df['ML_Prob'] >= 0.5).astype(int)

# Repeated Stratified K-Fold for hybrid evaluation
rskf_eval = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
hybrid_accuracies = []

fold_num = 0
for train_idx, val_idx in rskf_eval.split(train_df, train_df['Survived']):
    fold_num += 1
    tr_set = train_df.iloc[train_idx]
    val_set = train_df.iloc[val_idx].copy()
    
    tr_wc = tr_set[tr_set['IsWomanOrChild'] == 1]
    group_wc_survival = tr_wc.groupby('GroupId')['Survived'].agg(['count', 'mean'])
    
    val_preds = []
    for idx, row in val_set.iterrows():
        group_id = row['GroupId']
        is_wc = row['IsWomanOrChild']
        ml_pred = row['ML_Pred']
        
        pred = ml_pred
        
        if group_id in group_wc_survival.index:
            count = group_wc_survival.loc[group_id, 'count']
            mean_survival = group_wc_survival.loc[group_id, 'mean']
            
            if count > 0:
                if is_wc:
                    if mean_survival == 0.0:
                        pred = 0
                else:
                    if mean_survival == 1.0:
                        pred = 1
                        
        val_preds.append(pred)
        
    acc = accuracy_score(val_set['Survived'], val_preds)
    hybrid_accuracies.append(acc)

with open("hybrid_output_utf8.txt", "w", encoding="utf-8") as f:
    f.write("=== Model CV Results ===\n")
    for name in models:
        acc = accuracy_score(y_train_arr, (oof_probs[name] >= 0.5).astype(int))
        f.write(f"{name} OOF Accuracy: {acc:.5f}\n")
        
    f.write(f"\nBest Ensemble ML-only OOF Accuracy: {best_acc:.5f}\n")
    f.write("Weights:\n")
    for name, w in zip(model_names, best_weights):
        f.write(f"  {name}: {w:.4f}\n")
        
    f.write(f"\nHybrid OOF Accuracy (across folds): {np.mean(hybrid_accuracies):.5f}\n")
    
    # Let's count how many corrections the hybrid model makes over the ML-only predictions
    # globally to see how often it triggers:
    total_group_wc = train_df[train_df['IsWomanOrChild'] == 1]
    group_wc_survival_global = total_group_wc.groupby('GroupId')['Survived'].agg(['count', 'mean'])
    
    corrections_applied = 0
    wc_died_corrections = 0
    men_survived_corrections = 0
    
    for idx, row in train_df.iterrows():
        group_id = row['GroupId']
        is_wc = row['IsWomanOrChild']
        ml_pred = row['ML_Pred']
        actual = row['Survived']
        
        # Exclude self
        other_members = train_df[(train_df['GroupId'] == group_id) & (train_df.index != idx)]
        other_wc = other_members[other_members['IsWomanOrChild'] == 1]
        
        if len(other_wc) > 0:
            mean_survival = other_wc['Survived'].mean()
            if is_wc and mean_survival == 0.0:
                if ml_pred == 1:
                    corrections_applied += 1
                    wc_died_corrections += 1
            elif not is_wc and mean_survival == 1.0:
                if ml_pred == 0:
                    corrections_applied += 1
                    men_survived_corrections += 1
                    
    f.write(f"\nTotal corrections over ML predictions (leave-one-out): {corrections_applied}\n")
    f.write(f"  WCs who were corrected to die: {wc_died_corrections}\n")
    f.write(f"  Men who were corrected to survive: {men_survived_corrections}\n")
