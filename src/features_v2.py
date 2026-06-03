"""
Titanic Feature Engineering V2 - Fixed Group Survival + Interaction Features
Key changes from V1:
  1. Group Survival uses surname+fare matching between train<->test (not leaking via OOF trick)
  2. Added Sex*Pclass interaction (the single most powerful feature)
  3. Ticket prefix extraction  
  4. Better cabin handling with HasCabin binary
  5. Post-prediction family consistency corrections
"""
import pandas as pd
import numpy as np


def extract_title(df):
    """Extracts title from the Name column and maps rare titles to broader categories."""
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


def impute_age(df, median_ages=None):
    """Imputes missing ages based on Pclass, Sex, and Title subgroups."""
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


def extract_ticket_prefix(ticket):
    """Extracts the alphabetic prefix from a ticket number."""
    parts = ticket.split()
    if len(parts) > 1:
        prefix = parts[0].replace('.', '').replace('/', '').upper()
        return prefix
    return 'NONE'


def build_family_survival_features(train_df, test_df):
    """
    Build family/group survival features by linking train and test passengers
    through shared surnames and ticket numbers.
    
    For test passengers: look up the KNOWN survival outcomes of their 
    family/group members in the training set.
    
    For train passengers: use leave-one-out from the training set only.
    
    This avoids the target leakage that inflated V1's CV score.
    """
    # Extract surnames
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df['Surname'] = train_df['Name'].apply(lambda x: x.split(',')[0].strip().lower())
    test_df['Surname'] = test_df['Name'].apply(lambda x: x.split(',')[0].strip().lower())
    
    # Combined for group size calculations only (no survival info leaked)
    combined = pd.concat([train_df, test_df], ignore_index=True)
    ticket_counts = combined['Ticket'].value_counts().to_dict()
    combined['TicketGroupSize'] = combined['Ticket'].map(ticket_counts)
    
    train_size = len(train_df)
    train_df['TicketGroupSize'] = combined.iloc[:train_size]['TicketGroupSize'].values
    test_df['TicketGroupSize'] = combined.iloc[train_size:]['TicketGroupSize'].values
    
    # Family Size
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
    
    # --- Build survival lookup dictionaries from TRAINING data only ---
    
    # 1. Ticket-based survival lookup
    ticket_survival = {}
    for ticket, group in train_df.groupby('Ticket'):
        if len(group) > 1:
            for idx, row in group.iterrows():
                others = group.drop(idx)
                if ticket not in ticket_survival:
                    ticket_survival[ticket] = {}
                ticket_survival[ticket][row['PassengerId']] = others['Survived'].mean()
    
    # 2. Surname-based survival lookup (for families)
    surname_survival = {}
    for surname, group in train_df.groupby('Surname'):
        if len(group) > 1 and surname != '':
            for idx, row in group.iterrows():
                others = group.drop(idx)
                if surname not in surname_survival:
                    surname_survival[surname] = {}
                surname_survival[surname][row['PassengerId']] = others['Survived'].mean()
    
    # 3. Global ticket survival (all members, for test set lookup)
    ticket_survival_all = {}
    for ticket, group in train_df.groupby('Ticket'):
        ticket_survival_all[ticket] = group['Survived'].mean()
    
    surname_survival_all = {}
    for surname, group in train_df.groupby('Surname'):
        if len(group) > 1 and surname != '':
            surname_survival_all[surname] = group['Survived'].mean()
    
    # --- Assign features ---
    
    # For TRAIN: leave-one-out (exclude self)
    def get_train_group_survival(row):
        pid = row['PassengerId']
        ticket = row['Ticket']
        surname = row['Surname']
        
        # Try ticket first
        if ticket in ticket_survival and pid in ticket_survival[ticket]:
            return ticket_survival[ticket][pid]
        # Try surname
        if surname in surname_survival and pid in surname_survival[surname]:
            return surname_survival[surname][pid]
        return 0.5  # neutral default
    
    # For TEST: use all training members' outcomes
    def get_test_group_survival(row):
        ticket = row['Ticket']
        surname = row['Surname']
        
        # Try ticket first (strongest signal)
        if ticket in ticket_survival_all:
            return ticket_survival_all[ticket]
        # Try surname
        if surname in surname_survival_all:
            return surname_survival_all[surname]
        return 0.5  # neutral default
    
    train_df['GroupSurvival'] = train_df.apply(get_train_group_survival, axis=1)
    test_df['GroupSurvival'] = test_df.apply(get_test_group_survival, axis=1)
    
    return train_df, test_df


def engineer_features(train_path, test_path):
    """Complete feature engineering pipeline V2."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    target = train['Survived'].copy()
    train_df = train.copy()
    test_df = test.copy()
    
    # --- Title ---
    train_df = extract_title(train_df)
    test_df = extract_title(test_df)
    
    # --- Age Imputation ---
    train_df, median_ages = impute_age(train_df)
    test_df, _ = impute_age(test_df, median_ages)
    
    # --- Fare Imputation ---
    fare_median = train_df.groupby('Pclass')['Fare'].median().to_dict()
    train_df['Fare'] = train_df.apply(
        lambda r: fare_median.get(r['Pclass'], train_df['Fare'].median()) if pd.isna(r['Fare']) else r['Fare'], axis=1)
    test_df['Fare'] = test_df.apply(
        lambda r: fare_median.get(r['Pclass'], train_df['Fare'].median()) if pd.isna(r['Fare']) else r['Fare'], axis=1)
    
    # --- Embarked ---
    embarked_mode = train_df['Embarked'].mode()[0]
    train_df['Embarked'] = train_df['Embarked'].fillna(embarked_mode)
    test_df['Embarked'] = test_df['Embarked'].fillna(embarked_mode)
    
    # --- Deck from Cabin ---
    train_df['HasCabin'] = train_df['Cabin'].notna().astype(int)
    test_df['HasCabin'] = test_df['Cabin'].notna().astype(int)
    train_df['Deck'] = train_df['Cabin'].fillna('U').apply(lambda x: x[0]).replace('T', 'U')
    test_df['Deck'] = test_df['Cabin'].fillna('U').apply(lambda x: x[0]).replace('T', 'U')
    
    # --- Ticket Prefix ---
    train_df['TicketPrefix'] = train_df['Ticket'].apply(extract_ticket_prefix)
    test_df['TicketPrefix'] = test_df['Ticket'].apply(extract_ticket_prefix)
    
    # Keep only common prefixes, map rare ones to 'RARE'
    prefix_counts = train_df['TicketPrefix'].value_counts()
    common_prefixes = prefix_counts[prefix_counts >= 5].index.tolist()
    train_df['TicketPrefix'] = train_df['TicketPrefix'].apply(lambda x: x if x in common_prefixes else 'RARE')
    test_df['TicketPrefix'] = test_df['TicketPrefix'].apply(lambda x: x if x in common_prefixes else 'RARE')
    
    # --- Family & Group Survival ---
    train_df, test_df = build_family_survival_features(train_df, test_df)
    
    # --- Derived Features ---
    # Log Fare
    train_df['LogFare'] = np.log1p(train_df['Fare'])
    test_df['LogFare'] = np.log1p(test_df['Fare'])
    
    # Per-person fare
    train_df['FarePerPerson'] = train_df['Fare'] / train_df['TicketGroupSize']
    test_df['FarePerPerson'] = test_df['Fare'] / test_df['TicketGroupSize']
    train_df['LogFarePerPerson'] = np.log1p(train_df['FarePerPerson'])
    test_df['LogFarePerPerson'] = np.log1p(test_df['FarePerPerson'])
    
    # Age categories
    train_df['IsChild'] = (train_df['Age'] < 14).astype(int)
    test_df['IsChild'] = (test_df['Age'] < 14).astype(int)
    train_df['IsYoungAdult'] = ((train_df['Age'] >= 14) & (train_df['Age'] < 30)).astype(int)
    test_df['IsYoungAdult'] = ((test_df['Age'] >= 14) & (test_df['Age'] < 30)).astype(int)
    
    # Family features
    train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)
    test_df['IsAlone'] = (test_df['FamilySize'] == 1).astype(int)
    train_df['SmallFamily'] = ((train_df['FamilySize'] >= 2) & (train_df['FamilySize'] <= 4)).astype(int)
    test_df['SmallFamily'] = ((test_df['FamilySize'] >= 2) & (test_df['FamilySize'] <= 4)).astype(int)
    
    # *** KEY INTERACTION: Sex x Pclass ***
    # This is the single most powerful feature for Titanic
    train_df['Sex_Pclass'] = train_df['Sex'].map({'male': 0, 'female': 1}) * 3 + train_df['Pclass']
    test_df['Sex_Pclass'] = test_df['Sex'].map({'male': 0, 'female': 1}) * 3 + test_df['Pclass']
    
    # Age x Pclass interaction
    train_df['Age_Pclass'] = train_df['Age'] * train_df['Pclass']
    test_df['Age_Pclass'] = test_df['Age'] * test_df['Pclass']
    
    # --- Select final features ---
    features = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
        'LogFare', 'LogFarePerPerson',
        'Embarked', 'Title', 'Deck', 'TicketPrefix',
        'FamilySize', 'TicketGroupSize',
        'GroupSurvival',
        'HasCabin', 'IsChild', 'IsYoungAdult', 'IsAlone', 'SmallFamily',
        'Sex_Pclass', 'Age_Pclass'
    ]
    
    X_train = train_df[features].copy()
    y_train = target.copy()
    X_test = test_df[features].copy()
    passenger_ids = test_df['PassengerId'].copy()
    
    # Also return full dataframes for post-processing corrections
    return X_train, y_train, X_test, passenger_ids, train_df, test_df


if __name__ == "__main__":
    X_train, y_train, X_test, pids, _, _ = engineer_features("data/train.csv", "data/test.csv")
    print("V2 Feature engineering complete!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"\nFeatures: {list(X_train.columns)}")
    print(f"\nGroupSurvival distribution (train):\n{X_train['GroupSurvival'].describe()}")
    print(f"\nGroupSurvival distribution (test):\n{X_test['GroupSurvival'].describe()}")
    print(f"\nSex_Pclass value counts (train):\n{X_train['Sex_Pclass'].value_counts().sort_index()}")
