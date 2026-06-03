import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier

from features_v2 import engineer_features

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

def train_and_predict_v6():
    X_train, y_train, X_test, passenger_ids, train_df_full, test_df_full = engineer_features("data/train.csv", "data/test.csv")
    
    cat_cols = ['Sex', 'Embarked', 'Title', 'Deck', 'TicketPrefix']
    num_cols = ['Age', 'SibSp', 'Parch', 'LogFare', 'LogFarePerPerson',
                'FamilySize', 'TicketGroupSize', 'GroupSurvival',
                'HasCabin', 'IsChild', 'IsYoungAdult', 'IsAlone', 'SmallFamily',
                'Sex_Pclass', 'Age_Pclass']
                
    preprocessor = get_preprocessor(cat_cols, num_cols)
    
    # Exact V2 models
    estimators = [
        ('LR', LogisticRegression(C=0.8, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)),
        ('RF', RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_split=8, min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1)),
        ('GBM', GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, min_samples_split=10, min_samples_leaf=5, random_state=42)),
        ('SVC', SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42)),
        ('XGB', XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5, random_state=42, n_jobs=-1, eval_metric='logloss')),
        ('LGBM', LGBMClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0, min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1)),
        ('CatBoost', CatBoostClassifier(iterations=200, depth=3, learning_rate=0.05, l2_leaf_reg=5, min_data_in_leaf=10, random_seed=42, verbose=0, thread_count=-1))
    ]
    
    # Use Stacking instead of random Dirichlet weights
    # Meta-learner is Logistic Regression
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=0.1, random_state=42),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=1,
        passthrough=False
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('stack', stack)
    ])
    
    print("Training Stacking Classifier...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on training set
    train_preds = pipeline.predict(X_train)
    acc = accuracy_score(y_train, train_preds)
    print(f"Stacking Training Accuracy: {acc:.5f}")
    
    # Predict test set
    final_preds = pipeline.predict(X_test)
    
    # Apply V2 Family Consistency Corrections (crucial for V2's high score)
    print("\nApplying V2 Family Consistency Corrections...")
    from train_v2 import apply_family_corrections
    final_preds = apply_family_corrections(test_df_full, final_preds, passenger_ids.values)
    
    # Save submission
    os.makedirs("submissions", exist_ok=True)
    submission = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': final_preds})
    submission_path = "submissions/submission_v6.csv"
    submission.to_csv(submission_path, index=False)
    
    import zipfile
    zip_path = submission_path.replace(".csv", ".zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(submission_path, os.path.basename(submission_path))
        
    print(f"\nSubmission saved: {submission_path}")
    print(f"Total rows: {len(submission)}")
    print(f"Class distribution:\n{submission['Survived'].value_counts(normalize=True).round(4)}")
    
    # Compare with V2
    v2 = pd.read_csv("submissions/submission_v2.csv")
    diffs = (submission['Survived'].values != v2['Survived'].values).sum()
    print(f"\nPredictions different from V2: {diffs}")

if __name__ == "__main__":
    train_and_predict_v6()
