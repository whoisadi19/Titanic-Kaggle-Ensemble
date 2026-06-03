import pandas as pd
import numpy as np

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Extract Surname
train['Surname'] = train['Name'].apply(lambda x: x.split(',')[0].strip().lower())
test['Surname'] = test['Name'].apply(lambda x: x.split(',')[0].strip().lower())

test_tickets = set(test['Ticket'])
train_tickets = set(train['Ticket'])
shared_tickets = test_tickets.intersection(train_tickets)

combined = pd.concat([train, test], ignore_index=True)
combined['Surname'] = combined['Name'].apply(lambda x: x.split(',')[0].strip().lower())

ticket_counts = combined['Ticket'].value_counts()
combined['TicketGroupSize'] = combined['Ticket'].map(ticket_counts)

# Define Woman or Child (female or Master title or Age < 14)
combined['IsWomanOrChild'] = ((combined['Sex'] == 'female') | (combined['Name'].str.contains('Master.')) | (combined['Age'] < 14)).astype(int)

train_wc = combined[combined['Survived'].notna() & (combined['IsWomanOrChild'] == 1)]
ticket_wc_survival = train_wc.groupby('Ticket')['Survived'].agg(['count', 'mean', 'sum'])

test_df = combined[combined['Survived'].isna()].copy()
test_df['Ticket_WC_Count'] = test_df['Ticket'].map(ticket_wc_survival['count']).fillna(0)
test_df['Ticket_WC_Mean'] = test_df['Ticket'].map(ticket_wc_survival['mean']).fillna(np.nan)

train_wc_surname = train_wc.groupby('Surname')['Survived'].agg(['count', 'mean', 'sum'])
test_df['Surname_WC_Count'] = test_df['Surname'].map(train_wc_surname['count']).fillna(0)
test_df['Surname_WC_Mean'] = test_df['Surname'].map(train_wc_surname['mean']).fillna(np.nan)

with open("explore_output_utf8.txt", "w", encoding="utf-8") as f:
    f.write(f"Total test rows: {len(test)}\n")
    f.write(f"Shared tickets between train and test: {len(shared_tickets)}\n")
    f.write(f"Number of test passengers with tickets also in train: {len(test[test['Ticket'].isin(shared_tickets)])}\n")
    
    test_surnames = set(test['Surname'])
    train_surnames = set(train['Surname'])
    shared_surnames = test_surnames.intersection(train_surnames)
    f.write(f"Shared surnames: {len(shared_surnames)}\n")
    
    f.write("\nTicket WC Survival in Train (First 10):\n")
    f.write(ticket_wc_survival.head(10).to_string())
    f.write("\n\nTest passengers with known ticket WC survival count:\n")
    f.write(test_df['Ticket_WC_Count'].value_counts().to_string())
    
    f.write("\n\nTest passengers with known surname WC survival count:\n")
    f.write(test_df['Surname_WC_Count'].value_counts().to_string())
    
    f.write("\n\nLet's analyze overlapping ticket group survival outcomes in training:\n")
    # For tickets shared between train and test:
    # Let's print the ticket, the members in train and their survival, and the members in test.
    for ticket in list(shared_tickets)[:15]:
        train_members = train[train['Ticket'] == ticket]
        test_members = test[test['Ticket'] == ticket]
        f.write(f"\nTicket: {ticket}\n")
        f.write("  Train members:\n")
        f.write(train_members[['Name', 'Sex', 'Age', 'Survived']].to_string())
        f.write("\n  Test members:\n")
        f.write(test_members[['Name', 'Sex', 'Age']].to_string())
        f.write("\n" + "-"*40 + "\n")
