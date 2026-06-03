import pandas as pd
import numpy as np

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Combine for grouping
combined = pd.concat([train, test], ignore_index=True)
combined['Surname'] = combined['Name'].apply(lambda x: x.split(',')[0].strip())

# Define Woman and Boy
combined['Boy'] = combined['Name'].str.contains('Master').astype(int)
combined['Woman'] = (combined['Sex'] == 'female').astype(int)
combined['WomanOrBoy'] = (combined['Boy'] | combined['Woman']).astype(int)

# 1. Group by Surname first
surname_survival = {}
for surname, group in combined.groupby('Surname'):
    # Only care about women and boys in the group who are in the training set
    train_group = group[group['Survived'].notna()]
    train_wb = train_group[train_group['WomanOrBoy'] == 1]
    
    if len(train_wb) > 0:
        surname_survival[surname] = train_wb['Survived'].mean()

# 2. Group by Ticket
ticket_survival = {}
for ticket, group in combined.groupby('Ticket'):
    train_group = group[group['Survived'].notna()]
    train_wb = train_group[train_group['WomanOrBoy'] == 1]
    
    if len(train_wb) > 0:
        ticket_survival[ticket] = train_wb['Survived'].mean()

# Generate predictions for test set
test_preds = []
corrections_wb_die = 0
corrections_man_survive = 0

for idx, row in combined[combined['Survived'].isna()].iterrows():
    is_wb = row['WomanOrBoy'] == 1
    surname = row['Surname']
    ticket = row['Ticket']
    
    # Base prediction: Women and Boys survive, Men die
    pred = 1 if is_wb else 0
    
    # Try to find group survival rate (Ticket preferred over Surname)
    group_rate = None
    if ticket in ticket_survival:
        group_rate = ticket_survival[ticket]
    elif surname in surname_survival:
        group_rate = surname_survival[surname]
        
    # Apply Chris Deotte rules
    if group_rate is not None:
        if is_wb and group_rate == 0.0:
            pred = 0  # Woman/Boy dies if all train W/B in group died
            corrections_wb_die += 1
        elif not is_wb and group_rate == 1.0:
            pred = 1  # Man survives if all train W/B in group survived
            corrections_man_survive += 1
            
    test_preds.append(pred)

print(f"Woman/Boy flipped to DIE: {corrections_wb_die}")
print(f"Man flipped to SURVIVE: {corrections_man_survive}")

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_preds
})
submission.to_csv("submissions/submission_wcg_only.csv", index=False)
print("Saved to submissions/submission_wcg_only.csv")
