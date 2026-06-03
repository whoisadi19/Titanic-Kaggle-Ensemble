"""
Titanic Feature Engineering V3 - Advanced Age Imputation

Changes from V2:
1. Imputes missing Age using a RandomForestRegressor trained on 
   Pclass, Title, Sex, SibSp, Parch, and Fare, rather than simple group medians.
2. The hope is that this uncovers a few hidden "children" that the median 
   approach misclassified as adults.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

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

def impute_age_ml(train_df, test_df):
    """
    Uses a RandomForestRegressor to predict missing ages.
    We combine train and test to get more data for training the regressor,
    but we only use rows with non-null Age to train.
    """
    combined = pd.concat([train_df, test_df], ignore_index=True)
    
    # Features for age prediction
    age_features = ['Pclass', 'Sex', 'Title', 'SibSp', 'Parch', 'Fare']
    
    # Preprocess
    # Fill missing fare for the 1 test set passenger
    combined['Fare'] = combined['Fare'].fillna(combined['Fare'].median())
    
    # One-hot encode Sex and Title
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    cat_feats = pd.DataFrame(encoder.fit_transform(combined[['Sex', 'Title']]))
    cat_feats.columns = encoder.get_feature_names_out(['Sex', 'Title'])
    
    # Prepare X
    X_all = pd.concat([combined[['Pclass', 'SibSp', 'Parch', 'Fare']], cat_feats], axis=1)
    
    # Split into train (has age) and test (needs age)
    has_age = combined['Age'].notna()
    needs_age = combined['Age'].isna()
    
    X_train_age = X_all[has_age]
    y_train_age = combined.loc[has_age, 'Age']
    X_test_age = X_all[needs_age]
    
    if len(X_test_age) > 0:
        rf = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=7)
        rf.fit(X_train_age, y_train_age)
        predicted_ages = rf.predict(X_test_age)
        
        # Apply predictions back to combined
        combined.loc[needs_age, 'Age'] = predicted_ages
    
    # Split back to train and test
    train_imputed = combined.iloc[:len(train_df)].copy()
    test_imputed = combined.iloc[len(train_df):].copy()
    
    return train_imputed, test_imputed

def extract_ticket_prefix(ticket):
    parts = ticket.split()
    if len(parts) > 1:
        prefix = parts[0].replace('.', '').replace('/', '').upper()
        return prefix
    return 'NONE'

def engineer_features_v3(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    passenger_ids = test_df['PassengerId']
    
    train_df = extract_title(train_df)
    test_df = extract_title(test_df)
    
    # Advanced Age Imputation
    train_df, test_df = impute_age_ml(train_df, test_df)
    
    combined = pd.concat([train_df, test_df], ignore_index=True)
    
    # Fare Imputation
    combined['Fare'] = combined['Fare'].fillna(combined['Fare'].median())
    combined['LogFare'] = np.log1p(combined['Fare'])
    
    # Family and Group Size
    combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
    ticket_counts = combined['Ticket'].value_counts()
    combined['TicketGroupSize'] = combined['Ticket'].map(ticket_counts)
    
    combined['FarePerPerson'] = combined['Fare'] / combined['TicketGroupSize']
    combined['LogFarePerPerson'] = np.log1p(combined['FarePerPerson'])
    
    # Is Alone / Small Family
    combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)
    combined['SmallFamily'] = ((combined['FamilySize'] > 1) & (combined['FamilySize'] < 5)).astype(int)
    
    # Binned Features based on our highly accurate ML-imputed age
    combined['IsChild'] = (combined['Age'] < 14).astype(int)
    combined['IsYoungAdult'] = ((combined['Age'] >= 14) & (combined['Age'] < 30)).astype(int)
    
    # Extract Cabin Deck
    combined['HasCabin'] = combined['Cabin'].notna().astype(int)
    combined['Deck'] = combined['Cabin'].apply(lambda s: str(s)[0] if pd.notnull(s) else 'U')
    
    # Ticket Prefix
    combined['TicketPrefix'] = combined['Ticket'].apply(extract_ticket_prefix)
    
    # Interactions
    combined['Sex_Pclass'] = combined['Sex'].map({'female': 1, 'male': 0}) * 3 + combined['Pclass']
    combined['Age_Pclass'] = combined['Age'] * combined['Pclass']
    
    # Group Survival Setup
    combined['Surname'] = combined['Name'].apply(lambda x: x.split(',')[0].strip())
    
    # Split back
    train_df = combined.iloc[:len(train_df)].copy()
    test_df = combined.iloc[len(train_df):].copy()
    
    # Group Survival Extraction
    ticket_survival = {}
    for ticket, group in train_df.groupby('Ticket'):
        if len(group) > 1:
            for idx, row in group.iterrows():
                others = group.drop(idx)
                if ticket not in ticket_survival:
                    ticket_survival[ticket] = {}
                ticket_survival[ticket][row['PassengerId']] = others['Survived'].mean()
                
    surname_survival = {}
    for surname, group in train_df.groupby('Surname'):
        if len(group) > 1 and surname != '':
            for idx, row in group.iterrows():
                others = group.drop(idx)
                if surname not in surname_survival:
                    surname_survival[surname] = {}
                surname_survival[surname][row['PassengerId']] = others['Survived'].mean()
                
    ticket_survival_all = {}
    for ticket, group in train_df.groupby('Ticket'):
        ticket_survival_all[ticket] = group['Survived'].mean()
        
    surname_survival_all = {}
    for surname, group in train_df.groupby('Surname'):
        if len(group) > 1 and surname != '':
            surname_survival_all[surname] = group['Survived'].mean()
            
    def get_train_group_survival(row):
        pid = row['PassengerId']
        ticket = row['Ticket']
        surname = row['Surname']
        if ticket in ticket_survival and pid in ticket_survival[ticket]:
            return ticket_survival[ticket][pid]
        if surname in surname_survival and pid in surname_survival[surname]:
            return surname_survival[surname][pid]
        return 0.5
        
    def get_test_group_survival(row):
        ticket = row['Ticket']
        surname = row['Surname']
        if ticket in ticket_survival_all:
            return ticket_survival_all[ticket]
        if surname in surname_survival_all:
            return surname_survival_all[surname]
        return 0.5
        
    train_df['GroupSurvival'] = train_df.apply(get_train_group_survival, axis=1)
    test_df['GroupSurvival'] = test_df.apply(get_test_group_survival, axis=1)
    
    # Prepare features for ML
    features = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'LogFare', 'LogFarePerPerson',
        'Embarked', 'Title', 'Deck', 'TicketPrefix', 'FamilySize', 'TicketGroupSize',
        'IsAlone', 'SmallFamily', 'IsChild', 'IsYoungAdult', 'HasCabin',
        'Sex_Pclass', 'Age_Pclass', 'GroupSurvival'
    ]
    
    train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
    test_df['Embarked'] = test_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
    
    X_train = train_df[features]
    y_train = train_df['Survived']
    X_test = test_df[features]
    
    return X_train, y_train, X_test, passenger_ids, train_df, test_df
