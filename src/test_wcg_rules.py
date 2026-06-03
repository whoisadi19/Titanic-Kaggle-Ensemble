import pandas as pd
import numpy as np

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Combine train and test to get all passengers and surnames
combined = pd.concat([train, test], ignore_index=True)
combined['Surname'] = combined['Name'].apply(lambda x: x.split(',')[0].strip().lower())

# Define Woman or Child (female or Master title or Age < 14)
combined['IsWomanOrChild'] = ((combined['Sex'] == 'female') | (combined['Name'].str.contains('Master.')) | (combined['Age'] < 14)).astype(int)

def get_group_id(row):
    ticket = row['Ticket']
    if ticket_counts[ticket] > 1:
        return f"ticket_{ticket}"
    
    surname = row['Surname']
    pclass = row['Pclass']
    embarked = row['Embarked']
    return f"family_{surname}_{pclass}_{embarked}"

ticket_counts = combined['Ticket'].value_counts()
combined['GroupId'] = combined.apply(get_group_id, axis=1)

train_df = combined[combined['Survived'].notna()].copy()
test_df = combined[combined['Survived'].isna()].copy()

oof_preds = []
targets = []

for idx, row in train_df.iterrows():
    group_id = row['GroupId']
    is_wc = row['IsWomanOrChild']
    
    # Base prediction
    if is_wc:
        pred = 1
    else:
        pred = 0
        
    other_members = train_df[(train_df['GroupId'] == group_id) & (train_df.index != idx)]
    other_wc = other_members[other_members['IsWomanOrChild'] == 1]
    
    if len(other_wc) > 0:
        mean_survival = other_wc['Survived'].mean()
        if is_wc:
            if mean_survival == 0.0:
                pred = 0
        else:
            if mean_survival == 1.0:
                pred = 1
                
    oof_preds.append(pred)
    targets.append(row['Survived'])

train_df['WCG_Pred'] = oof_preds
acc = (train_df['WCG_Pred'] == train_df['Survived']).mean()

with open("wcg_output_utf8.txt", "w", encoding="utf-8") as f:
    f.write(f"Pure WCG Rule OOF Accuracy on train: {acc:.5f}\n\n")
    errors = train_df[train_df['WCG_Pred'] != train_df['Survived']]
    f.write(f"Number of errors: {len(errors)}\n\n")
    f.write("Errors by Sex and Pclass:\n")
    f.write(errors.groupby(['Sex', 'Pclass'])['Survived'].value_counts().to_string())
    f.write("\n\nLet's check the survival rate of alone passengers (group size = 1) in training:\n")
    alone_train = train_df[train_df['GroupId'].map(combined['GroupId'].value_counts()) == 1]
    f.write(f"Number of alone training passengers: {len(alone_train)}\n")
    f.write(alone_train.groupby(['Sex', 'Pclass'])['Survived'].agg(['count', 'mean']).to_string())
