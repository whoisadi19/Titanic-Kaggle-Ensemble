import pandas as pd
import numpy as np

def train_and_predict_v10():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    
    # Save passenger IDs for submission
    passenger_ids = test['PassengerId'].values
    
    # Combine datasets
    full = pd.concat([train, test], ignore_index=True)
    
    # 1. Name Features
    full['Title'] = full['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    full['Surname'] = full['Name'].apply(lambda x: x.split(',')[0].strip())
    
    # Title consolidation
    full['Title'] = full['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss')
    full['Title'] = full['Title'].replace(['Don', 'Rev', 'Dr', 'Major', 'Sir', 'Col', 'Capt', 'Jonkheer'], 'Mr')
    
    # 2. Family Size
    full['FamilySize'] = full['SibSp'] + full['Parch'] + 1
    
    # 3. Family Survival Feature (The key to >0.80)
    full['Family_Survival'] = 0.5
    
    # Group by Surname and Fare to ensure actual families
    for _, grp in full.groupby(['Surname', 'Fare']):
        if len(grp) > 1:
            for idx, row in grp.iterrows():
                smax = grp.drop(idx)['Survived'].max()
                smin = grp.drop(idx)['Survived'].min()
                passID = row['PassengerId']
                if smax == 1.0:
                    full.loc[full['PassengerId'] == passID, 'Family_Survival'] = 1
                elif smin == 0.0:
                    full.loc[full['PassengerId'] == passID, 'Family_Survival'] = 0
                    
    # Group by Ticket to catch friends/nannies
    for _, grp in full.groupby('Ticket'):
        if len(grp) > 1:
            for idx, row in grp.iterrows():
                # If Family_Survival is still neutral or 0, check Ticket
                if full.loc[full['PassengerId'] == row['PassengerId'], 'Family_Survival'].values[0] in [0.5, 0]:
                    smax = grp.drop(idx)['Survived'].max()
                    smin = grp.drop(idx)['Survived'].min()
                    passID = row['PassengerId']
                    if smax == 1.0:
                        full.loc[full['PassengerId'] == passID, 'Family_Survival'] = 1
                    elif smin == 0.0:
                        full.loc[full['PassengerId'] == passID, 'Family_Survival'] = 0
    
    # Split back to train and test
    train_df = full.iloc[:len(train)].copy()
    test_df = full.iloc[len(train):].copy()
    
    # 4. Standard features
    train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
    test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
    
    features = ['Pclass', 'Sex', 'Family_Survival', 'FamilySize']
    
    X_train = train_df[features]
    y_train = train_df['Survived'].astype(int)
    X_test = test_df[features]
    
    # 5. Train model (Random Forest is sufficient here, the magic is in the features)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=3, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    # 6. Apply strict Chris Deotte rules as post-processing to ensure no ML fuzziness
    # Women and children survive, men die, UNLESS contradicted by Family_Survival
    for i, idx in enumerate(test_df.index):
        row = test_df.loc[idx]
        is_boy = (row['Title'] == 'Master')
        is_woman = (row['Sex'] == 1)
        
        if is_boy or is_woman:
            if row['Family_Survival'] == 0:
                predictions[i] = 0
            else:
                predictions[i] = 1
        else:
            if row['Family_Survival'] == 1:
                predictions[i] = 1
            else:
                predictions[i] = 0
                
    # Save submission
    import os
    os.makedirs("submissions", exist_ok=True)
    submission = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
    sub_path = "submissions/submission_v10.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Saved to {sub_path}")
    print(f"Total rows: {len(submission)}")
    print(f"Class distribution:\n{submission['Survived'].value_counts(normalize=True).round(4)}")

if __name__ == "__main__":
    train_and_predict_v10()
