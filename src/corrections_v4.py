"""
Titanic V4: Conservative corrections on top of V2 (our best score: 0.79665).

Key lesson from V3: Flipping males to "survive" because their wives survived was 
WRONG. On the Titanic, "women and children first" meant wives were saved but 
husbands DIED (e.g., Col. John Jacob Astor). 

V4 Strategy:
- Start from V2 predictions (0.79665 baseline)
- ONLY apply WC->0 corrections: women/children whose ENTIRE family group died
- Do NOT flip any males from 0->1
- Be very conservative: only correct when the evidence is overwhelming
"""
import os
import zipfile
import pandas as pd
import numpy as np


def apply_conservative_corrections(train_path, test_path, base_sub_path, output_sub_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    base_sub = pd.read_csv(base_sub_path)
    
    # Extract metadata
    train['Surname'] = train['Name'].apply(lambda x: x.split(',')[0].strip().lower())
    test['Surname'] = test['Name'].apply(lambda x: x.split(',')[0].strip().lower())
    
    def get_title(name):
        return name.split(',')[1].split('.')[0].strip()
    train['Title'] = train['Name'].apply(get_title)
    test['Title'] = test['Name'].apply(get_title)
    
    # Woman or Child: female, or age < 14, or title is Master
    train['IsWomanOrChild'] = ((train['Sex'] == 'female') | (train['Age'] < 14) | (train['Title'] == 'Master')).astype(int)
    test['IsWomanOrChild'] = ((test['Sex'] == 'female') | (test['Age'] < 14) | (test['Title'] == 'Master')).astype(int)
    
    # Merge base predictions into test
    test_df = test.merge(base_sub, on='PassengerId', how='left')
    
    # Build group identifiers using combined ticket counts
    combined = pd.concat([train, test], ignore_index=True)
    ticket_counts = combined['Ticket'].value_counts().to_dict()
    
    # Group survival lookup from training set - TICKET ONLY (most reliable)
    # We only use ticket-based groups, no surname fallback for corrections
    train_wc = train[train['IsWomanOrChild'] == 1]
    
    # For each ticket with multiple passengers, compute WC survival
    ticket_wc_survival = {}
    for ticket, group in train_wc.groupby('Ticket'):
        ticket_wc_survival[ticket] = {
            'count': len(group),
            'mean': group['Survived'].mean(),
            'all_died': group['Survived'].sum() == 0,
            'all_survived': group['Survived'].mean() == 1.0
        }
    
    # Also build surname-based lookup (but only for surnames that are unique enough)
    # Only use surnames that appear exactly once in training AND exactly once in test
    # to avoid false matches like "Kelly", "Andersson"
    surname_wc_survival = {}
    for (surname, pclass), group in train_wc.groupby(['Surname', 'Pclass']):
        if surname == '':
            continue
        # Only use if the surname+pclass combo is unique enough
        train_surname_count = len(train[train['Surname'] == surname])
        if train_surname_count <= 4:  # Small family, reliable signal
            surname_wc_survival[(surname, pclass)] = {
                'count': len(group),
                'mean': group['Survived'].mean(),
                'all_died': group['Survived'].sum() == 0,
                'all_survived': group['Survived'].mean() == 1.0
            }
    
    corrections = []
    final_predictions = test_df['Survived'].values.copy()
    
    print("=" * 60)
    print("  V4: Conservative Corrections (WC->0 ONLY, no male flips)")
    print("=" * 60)
    
    for i, row in test_df.iterrows():
        idx = i if isinstance(i, int) else test_df.index.get_loc(i)
        pid = row['PassengerId']
        name = row['Name']
        sex = row['Sex']
        pclass = row['Pclass']
        ticket = row['Ticket']
        is_wc = row['IsWomanOrChild']
        ml_pred = row['Survived']
        surname = row['Surname']
        
        # ONLY correct women/children predicted to survive -> die
        if not is_wc:
            continue  # Skip all males entirely
        if ml_pred == 0:
            continue  # Already predicted to die, no correction needed
            
        # Check ticket-based group first (strongest signal)
        corrected = False
        if ticket in ticket_wc_survival:
            info = ticket_wc_survival[ticket]
            if info['all_died'] and info['count'] >= 1:
                final_predictions[idx] = 0
                corrected = True
                corrections.append({
                    'PassengerId': pid, 'Name': name, 'Sex': sex, 'Pclass': pclass,
                    'Ticket': ticket, 'Original': ml_pred, 'Corrected': 0,
                    'Reason': f"All {info['count']} WCs on ticket {ticket} DIED in training"
                })
                print(f"  [WC->0] {name} (Pclass {pclass}, Ticket {ticket})")
                print(f"    All {info['count']} WCs on this ticket died in training")
        
        # Fallback: surname + pclass (only if ticket didn't match)
        if not corrected:
            key = (surname, pclass)
            if key in surname_wc_survival:
                info = surname_wc_survival[key]
                if info['all_died'] and info['count'] >= 2:  # Need at least 2 WCs dying
                    final_predictions[idx] = 0
                    corrected = True
                    corrections.append({
                        'PassengerId': pid, 'Name': name, 'Sex': sex, 'Pclass': pclass,
                        'Ticket': ticket, 'Original': ml_pred, 'Corrected': 0,
                        'Reason': f"All {info['count']} WCs with surname '{surname}' Pclass {pclass} DIED in training"
                    })
                    print(f"  [WC->0] {name} (Pclass {pclass}, Surname {surname})")
                    print(f"    All {info['count']} WCs with this surname+class died in training")
    
    print(f"\nTotal corrections applied: {len(corrections)}")
    
    # Save corrected submission
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': final_predictions
    })
    submission.to_csv(output_sub_path, index=False)
    print(f"Submission saved: {output_sub_path}")
    
    # Save corrections log
    if corrections:
        pd.DataFrame(corrections).to_csv("submissions/corrections_v4_log.csv", index=False)
    
    # Create zip
    zip_path = output_sub_path.replace(".csv", ".zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(output_sub_path, os.path.basename(output_sub_path))
    print(f"Zip archive: {zip_path}")
    
    # Distribution
    print(f"\nSurvival distribution:")
    print(f"  Died:     {(final_predictions == 0).sum()} ({(final_predictions == 0).mean():.4f})")
    print(f"  Survived: {(final_predictions == 1).sum()} ({(final_predictions == 1).mean():.4f})")
    
    # Show diff from base
    diffs = (final_predictions != test_df['Survived'].values).sum()
    print(f"\nTotal predictions changed from V2 base: {diffs}")


if __name__ == "__main__":
    apply_conservative_corrections(
        train_path="data/train.csv",
        test_path="data/test.csv",
        base_sub_path="submissions/submission_v2.csv",  # Our best: 0.79665
        output_sub_path="submissions/submission_v4.csv"
    )
