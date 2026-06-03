import pandas as pd
import numpy as np

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sub_v2 = pd.read_csv("submissions/submission_v2.csv")

# Combine train and test to identify groups
combined = pd.concat([train, test], ignore_index=True)
combined['Surname'] = combined['Name'].apply(lambda x: x.split(',')[0].strip().lower())

# Define group strictly by Ticket number
ticket_counts = combined['Ticket'].value_counts()
combined['IsWomanOrChild'] = ((combined['Sex'] == 'female') | (combined['Name'].str.contains('Master.')) | (combined['Age'] < 14)).astype(int)

# Split back
train_df = combined[combined['Survived'].notna()].copy()
test_df = combined[combined['Survived'].isna()].copy()

# Drop empty Survived column
test_df = test_df.drop(columns=['Survived'])
test_df = test_df.merge(sub_v2, on='PassengerId', how='left')

# Calculate group survival strictly by TICKET
train_wc = train_df[train_df['IsWomanOrChild'] == 1]
group_wc_survival = train_wc.groupby('Ticket')['Survived'].agg(['count', 'mean'])

# Let's inspect test set passengers and see what our ML model predicted vs what the training group survival says!
corrections = []

with open("corrections_ticket_only_utf8.txt", "w", encoding="utf-8") as f:
    f.write("=== Strict Ticket-Only Family/Group Analysis ===\n\n")

    for idx, row in test_df.iterrows():
        ticket = row['Ticket']
        is_wc = row['IsWomanOrChild']
        ml_pred = row['Survived']
        pid = row['PassengerId']
        name = row['Name']
        sex = row['Sex']
        pclass = row['Pclass']
        
        # Look up by Ticket only
        if ticket in group_wc_survival.index:
            wc_count = group_wc_survival.loc[ticket, 'count']
            wc_mean = group_wc_survival.loc[ticket, 'mean']
            
            if wc_count > 0:
                if is_wc:
                    # Woman/child in test set
                    # If all women/children in train died, she should probably die too
                    if wc_mean == 0.0:
                        if ml_pred == 1:
                            f.write(f"Correction suggested: Woman/Child {name} (Pclass {pclass}, Ticket {ticket}) predicted 1 -> change to 0 (all WCs in train group died)\n")
                            corrections.append((pid, 0))
                else:
                    # Adult male in test set
                    # If all women/children in train survived, he might have survived too
                    if wc_mean == 1.0:
                        if ml_pred == 0:
                            f.write(f"Correction suggested: Adult Male {name} (Pclass {pclass}, Ticket {ticket}) predicted 0 -> change to 1 (all WCs in train group survived)\n")
                            corrections.append((pid, 1))

    f.write(f"\nTotal corrections suggested: {len(corrections)}\n")
print(f"Done! Strict Ticket-Only corrections saved. Total suggestions: {len(corrections)}")
