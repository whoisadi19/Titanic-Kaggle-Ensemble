import os
import pandas as pd
import numpy as np

def extract_title(df):
    """Extracts title from the Name column and maps rare titles to broader categories."""
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    
    # Map title synonyms and rare titles
    title_mapping = {
        'Mlle': 'Miss',
        'Mme': 'Mrs',
        'Ms': 'Miss',
        'Lady': 'Royalty',
        'the Countess': 'Royalty',
        'Countess': 'Royalty',
        'Don': 'Royalty',
        'Dona': 'Royalty',
        'Sir': 'Royalty',
        'Jonkheer': 'Royalty',
        'Capt': 'Officer',
        'Col': 'Officer',
        'Major': 'Officer',
        'Dr': 'Officer',
        'Rev': 'Officer'
    }
    
    df['Title'] = df['Title'].replace(title_mapping)
    return df

def impute_age(df, median_ages=None):
    """
    Imputes missing ages based on Pclass, Sex, and Title.
    If median_ages dictionary is provided, uses it (for test set).
    Otherwise, calculates it from df (for train set) and returns the dictionary.
    """
    if median_ages is None:
        # Calculate median ages from train dataset
        median_ages = df.groupby(['Pclass', 'Sex', 'Title'])['Age'].median().to_dict()
        
        # Fallback 1: group by Sex and Pclass if combination is missing
        fallback_ages_1 = df.groupby(['Pclass', 'Sex'])['Age'].median().to_dict()
        
        # Fallback 2: overall median age
        fallback_overall = df['Age'].median()
    else:
        fallback_ages_1 = median_ages.get('fallback_1', {})
        fallback_overall = median_ages.get('fallback_overall', 28.0)

    # Impute missing ages
    def get_imputed_age(row):
        if not np.isnan(row['Age']):
            return row['Age']
        
        key = (row['Pclass'], row['Sex'], row['Title'])
        if key in median_ages and not np.isnan(median_ages[key]):
            return median_ages[key]
        
        key_fallback = (row['Pclass'], row['Sex'])
        if key_fallback in fallback_ages_1 and not np.isnan(fallback_ages_1[key_fallback]):
            return fallback_ages_1[key_fallback]
            
        return fallback_overall

    df['Age'] = df.apply(get_imputed_age, axis=1)
    
    if 'fallback_1' not in median_ages:
        median_ages['fallback_1'] = fallback_ages_1
        median_ages['fallback_overall'] = fallback_overall
        
    return df, median_ages

def extract_deck(df):
    """Extracts deck from Cabin and maps rare decks."""
    df['Cabin'] = df['Cabin'].fillna('U')
    df['Deck'] = df['Cabin'].apply(lambda x: x[0])
    
    # Map rare 'T' deck to 'U'
    df['Deck'] = df['Deck'].replace('T', 'U')
    return df

def process_family_and_groups(train_df, test_df):
    """
    Creates Family and Group survival features.
    Computes survival rates of group members (by Ticket & Surname) to capture group dynamics.
    Crucial: Computes these separately for Train and Test to avoid data leakage.
    """
    # Combine datasets temporarily for group size extraction
    combined = pd.concat([train_df, test_df], ignore_index=True)
    
    # Extract Surname
    combined['Surname'] = combined['Name'].apply(lambda x: x.split(',')[0].strip().lower())
    
    # Group Size by Ticket
    ticket_counts = combined['Ticket'].value_counts().to_dict()
    combined['GroupSize'] = combined['Ticket'].map(ticket_counts)
    
    # Family Size
    combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
    
    # Adjusted group size (max of ticket frequency and family size)
    combined['MaxGroupSize'] = combined[['GroupSize', 'FamilySize']].max(axis=1)
    
    # Re-split
    train_size = len(train_df)
    train_feat = combined.iloc[:train_size].copy()
    test_feat = combined.iloc[train_size:].copy()
    
    # Define Group ID based on Ticket
    # If Ticket is unique, but there's a family (FamilySize > 1), we use Surname
    train_feat['GroupID'] = train_feat['Ticket']
    test_feat['GroupID'] = test_feat['Ticket']
    
    # Calculate Group Survival Rates
    # We want to identify if other members in the same group survived or died.
    # Note: Only training set has the actual 'Survived' label.
    
    # Step 1: Calculate survival outcomes per group in train
    # Group Survival dictionary stores lists of (PassengerId, Survived) for each GroupID
    group_outcomes = {}
    for idx, row in train_feat.iterrows():
        gid = row['GroupID']
        pid = row['PassengerId']
        survived = row['Survived']
        if gid not in group_outcomes:
            group_outcomes[gid] = []
        group_outcomes[gid].append((pid, survived))
        
    def calculate_group_survival(row, is_train=True):
        gid = row['GroupID']
        pid = row['PassengerId']
        
        # If passenger is traveling alone (GroupSize == 1), survival rate is neutral (0.5)
        if row['MaxGroupSize'] <= 1:
            return 0.5
            
        # Check if group exists in training outcomes
        if gid not in group_outcomes:
            return 0.5
            
        outcomes = group_outcomes[gid]
        
        # Filter out the current passenger's own survival if in training set
        other_outcomes = [surv for p_id, surv in outcomes if not (is_train and p_id == pid)]
        
        if not other_outcomes:
            return 0.5
            
        # Group survival signal:
        # If any other female/child died in the group -> very negative signal
        # If any other male survived in the group -> very positive signal
        # Otherwise, average of other outcomes
        return np.mean(other_outcomes)

    train_feat['GroupSurvival'] = train_feat.apply(lambda r: calculate_group_survival(r, is_train=True), axis=1)
    test_feat['GroupSurvival'] = test_feat.apply(lambda r: calculate_group_survival(r, is_train=False), axis=1)
    
    return train_feat, test_feat

def engineer_features(train_path, test_path):
    """Runs the entire feature engineering pipeline and returns clean pandas DataFrames."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Drop Survived column for feature alignment (will re-add later)
    target = train['Survived']
    train_df = train.drop(columns=['Survived'])
    test_df = test.copy()
    
    # Extract Titles
    train_df = extract_title(train_df)
    test_df = extract_title(test_df)
    
    # Impute Age
    train_df, median_ages = impute_age(train_df)
    test_df, _ = impute_age(test_df, median_ages)
    
    # Impute missing Fare
    fare_median = train_df.groupby('Pclass')['Fare'].median().to_dict()
    train_df['Fare'] = train_df.apply(lambda r: fare_median[r['Pclass']] if np.isnan(r['Fare']) else r['Fare'], axis=1)
    test_df['Fare'] = test_df.apply(lambda r: fare_median[r['Pclass']] if np.isnan(r['Fare']) else r['Fare'], axis=1)
    
    # Extract Deck from Cabin
    train_df = extract_deck(train_df)
    test_df = extract_deck(test_df)
    
    # Fill missing Embarked with mode
    embarked_mode = train_df['Embarked'].mode()[0]
    train_df['Embarked'] = train_df['Embarked'].fillna(embarked_mode)
    test_df['Embarked'] = test_df['Embarked'].fillna(embarked_mode)
    
    # Add back Target for Group Survival calculations
    train_df['Survived'] = target
    
    # Process Family & Groups
    train_feat, test_feat = process_family_and_groups(train_df, test_df)
    
    # Calculate TicketFare (individual passenger fare)
    train_feat['TicketFare'] = train_feat['Fare'] / train_feat['GroupSize']
    test_feat['TicketFare'] = test_feat['Fare'] / test_feat['GroupSize']
    
    # Log Fare transforms
    train_feat['LogFare'] = np.log1p(train_feat['Fare'])
    test_feat['LogFare'] = np.log1p(test_feat['Fare'])
    train_feat['LogTicketFare'] = np.log1p(train_feat['TicketFare'])
    test_feat['LogTicketFare'] = np.log1p(test_feat['TicketFare'])
    
    # Age bins
    train_feat['IsChild'] = (train_feat['Age'] < 12).astype(int)
    test_feat['IsChild'] = (test_feat['Age'] < 12).astype(int)
    
    train_feat['IsSenior'] = (train_feat['Age'] > 60).astype(int)
    test_feat['IsSenior'] = (test_feat['Age'] > 60).astype(int)
    
    # Travel alone
    train_feat['IsAlone'] = (train_feat['FamilySize'] == 1).astype(int)
    test_feat['IsAlone'] = (test_feat['FamilySize'] == 1).astype(int)
    
    # Define categorical and numerical features to keep
    features_to_keep = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'LogFare', 'LogTicketFare',
        'Embarked', 'Title', 'Deck', 'FamilySize', 'MaxGroupSize',
        'GroupSurvival', 'IsChild', 'IsSenior', 'IsAlone'
    ]
    
    # Select columns
    X_train = train_feat[features_to_keep].copy()
    y_train = train_feat['Survived'].copy()
    X_test = test_feat[features_to_keep].copy()
    passenger_ids = test_feat['PassengerId'].copy()
    
    return X_train, y_train, X_test, passenger_ids

if __name__ == "__main__":
    X_train, y_train, X_test, passenger_ids = engineer_features("data/train.csv", "data/test.csv")
    print("Feature engineering complete!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(X_train.head())
