import pandas as pd
import numpy as np

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
test['Survived'] = np.nan
full = pd.concat([train, test])

# Extract Surname and Title
full['Surname'] = full['Name'].apply(lambda x: x.split(',')[0].strip())
full['Title'] = full['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

# Define Woman and Boy
full['Boy'] = (full['Title'] == 'Master').astype(int)
full['Woman'] = (full['Sex'] == 'female').astype(int)
full['WomanOrBoy'] = (full['Boy'] | full['Woman']).astype(int)

# Filter to only Woman and Boy in the training set
train_wb = full[(full['Survived'].notna()) & (full['WomanOrBoy'] == 1)]

# Calculate Survival Rates per Surname and Ticket based ONLY on Women/Boys in train
surname_survival = train_wb.groupby('Surname')['Survived'].mean().to_dict()
ticket_survival = train_wb.groupby('Ticket')['Survived'].mean().to_dict()

# Count frequencies in FULL dataset to avoid mapping solitary passengers
surname_counts = full['Surname'].value_counts().to_dict()
ticket_counts = full['Ticket'].value_counts().to_dict()

predictions = []
corrections_wb_die = 0
corrections_man_survive = 0

test_rows = full[full['Survived'].isna()]
for idx, row in test_rows.iterrows():
    is_wb = row['WomanOrBoy'] == 1
    surname = row['Surname']
    ticket = row['Ticket']
    
    # Base prediction
    pred = 1 if is_wb else 0
    
    # Find group rate, prioritizing Ticket over Surname
    group_rate = None
    
    # Only use Ticket if it's shared by more than 1 person in the whole dataset
    if ticket in ticket_survival and ticket_counts[ticket] > 1:
        group_rate = ticket_survival[ticket]
    # Otherwise use Surname, but ONLY if shared by more than 1 person in the whole dataset
    elif surname in surname_survival and surname_counts[surname] > 1:
        group_rate = surname_survival[surname]
        
    # Apply rules
    if group_rate is not None:
        if is_wb and group_rate == 0.0:
            pred = 0
            corrections_wb_die += 1
        elif not is_wb and group_rate == 1.0:
            pred = 1
            corrections_man_survive += 1
            
    predictions.append(pred)

print(f"Test women/boys flipped to die: {corrections_wb_die}")
print(f"Test men flipped to survive: {corrections_man_survive}")

submission = pd.DataFrame({
    'PassengerId': test_rows['PassengerId'],
    'Survived': predictions
})
submission.to_csv("submissions/submission_v8.csv", index=False)
print("Saved to submissions/submission_v8.csv")
