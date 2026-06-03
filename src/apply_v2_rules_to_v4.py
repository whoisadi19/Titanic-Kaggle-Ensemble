import pandas as pd
import numpy as np

def apply_family_corrections(test_df, predictions, passenger_ids):
    pred_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
    pred_df = pred_df.merge(
        test_df[['PassengerId', 'Surname', 'Ticket', 'Sex', 'Age', 'Pclass']],
        on='PassengerId', how='left'
    )
    
    corrections = 0
    
    # For passengers sharing a ticket, apply consistency
    for ticket, group in pred_df.groupby('Ticket'):
        if len(group) <= 1:
            continue
        
        females = group[group['Sex'] == 'female']
        males = group[group['Sex'] == 'male']
        
        # If ALL females in the group are predicted to die, males should also die
        if len(females) > 0 and females['Survived'].sum() == 0:
            for idx in males.index:
                if pred_df.loc[idx, 'Survived'] == 1:
                    print(f"  [Ticket] Changing male PID {pred_df.loc[idx, 'PassengerId']} to DIE because females died")
                    pred_df.loc[idx, 'Survived'] = 0
                    corrections += 1
        
        # If ALL males in the group survived, females should also survive
        if len(males) > 0 and males['Survived'].mean() == 1.0:
            for idx in females.index:
                if pred_df.loc[idx, 'Survived'] == 0:
                    print(f"  [Ticket] Changing female PID {pred_df.loc[idx, 'PassengerId']} to SURVIVE because males survived")
                    pred_df.loc[idx, 'Survived'] = 1
                    corrections += 1
    
    # For passengers sharing surname (family), apply consistency for women/children
    for surname, group in pred_df.groupby('Surname'):
        if len(group) <= 1:
            continue
        
        females = group[group['Sex'] == 'female']
        children = group[(group['Age'] < 14)]
        
        # If any female in the family died, children likely died too (family stayed together)
        if len(females) > 0 and females['Survived'].min() == 0 and len(children) > 0:
            for idx in children.index:
                if pred_df.loc[idx, 'Pclass'] == 3 and pred_df.loc[idx, 'Survived'] == 1:
                    print(f"  [Surname] Changing Pclass 3 child PID {pred_df.loc[idx, 'PassengerId']} to DIE because female in family died")
                    pred_df.loc[idx, 'Survived'] = 0
                    corrections += 1
                    
    print(f"Total corrections applied: {corrections}")
    return pred_df['Survived'].values

test_df = pd.read_csv("data/test.csv")
test_df['Surname'] = test_df['Name'].apply(lambda x: x.split(',')[0].strip().lower())

v4 = pd.read_csv("submissions/submission_v4.csv")
v4_preds = v4['Survived'].values
pids = v4['PassengerId'].values

print("Applying V2 corrections to V4 predictions...")
v5_preds = apply_family_corrections(test_df, v4_preds, pids)

v5 = pd.DataFrame({'PassengerId': pids, 'Survived': v5_preds})
v5.to_csv("submissions/submission_v5.csv", index=False)
print("Saved submission_v5.csv")

# How does V5 compare to V2?
v2 = pd.read_csv("submissions/submission_v2.csv")
diff = (v5['Survived'] != v2['Survived']).sum()
print(f"Differences between V5 and V2: {diff}")
