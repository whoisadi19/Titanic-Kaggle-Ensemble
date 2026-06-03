import os
import zipfile
import pandas as pd
import numpy as np


def apply_corrections(train_path, test_path, ml_sub_path, output_sub_path):
    """
    Applies strict ticket and surname-based family group corrections 
    to the machine learning ensemble predictions.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    ml_sub = pd.read_csv(ml_sub_path)
    
    # Extract Surnames
    train['Surname'] = train['Name'].apply(lambda x: x.split(',')[0].strip().lower())
    test['Surname'] = test['Name'].apply(lambda x: x.split(',')[0].strip().lower())
    
    # Extract Title for children verification
    def get_title(name):
        return name.split(',')[1].split('.')[0].strip()
    train['Title'] = train['Name'].apply(get_title)
    test['Title'] = test['Name'].apply(get_title)
    
    # Define Woman or Child (female, or age < 14, or title is Master)
    train['IsWomanOrChild'] = ((train['Sex'] == 'female') | (train['Age'] < 14) | (train['Title'] == 'Master')).astype(int)
    test['IsWomanOrChild'] = ((test['Sex'] == 'female') | (test['Age'] < 14) | (test['Title'] == 'Master')).astype(int)
    
    # Merge ML predictions into test dataframe
    test_df = test.copy()
    test_df = test_df.merge(ml_sub, on='PassengerId', how='left')
    
    # Compute group matching criteria
    combined = pd.concat([train, test], ignore_index=True)
    ticket_counts = combined['Ticket'].value_counts().to_dict()
    
    # Strict group definitions:
    # 1. Primary: Ticket number (count > 1 across train + test)
    # 2. Fallback: Surname + Pclass + Embarked
    train['GroupId'] = train.apply(
        lambda r: f"ticket_{r['Ticket']}" if ticket_counts[r['Ticket']] > 1 else f"family_{r['Surname']}_{r['Pclass']}_{r['Embarked']}",
        axis=1
    )
    test_df['GroupId'] = test_df.apply(
        lambda r: f"ticket_{r['Ticket']}" if ticket_counts[r['Ticket']] > 1 else f"family_{r['Surname']}_{r['Pclass']}_{r['Embarked']}",
        axis=1
    )
    
    # Group survival lookup in the training set
    train_wc = train[train['IsWomanOrChild'] == 1]
    group_wc_survival = train_wc.groupby('GroupId')['Survived'].agg(['count', 'mean'])
    
    corrections_applied = []
    
    # We will loop through the test set and check for corrections
    final_predictions = []
    
    print("\n=======================================================")
    print("      Applying Out-of-Fold Family Group Corrections   ")
    print("=======================================================\n")
    
    for idx, row in test_df.iterrows():
        pid = row['PassengerId']
        name = row['Name']
        sex = row['Sex']
        pclass = row['Pclass']
        ticket = row['Ticket']
        group_id = row['GroupId']
        is_wc = row['IsWomanOrChild']
        ml_pred = row['Survived']
        
        pred = ml_pred
        corrected = False
        reason = ""
        
        # Check if the passenger's group has women/children in the training set
        if group_id in group_wc_survival.index:
            wc_count = group_wc_survival.loc[group_id, 'count']
            wc_mean = group_wc_survival.loc[group_id, 'mean']
            
            if wc_count > 0:
                # If they are a woman/child, check if all WCs in the training group died
                if is_wc:
                    if wc_mean == 0.0:
                        if ml_pred == 1:
                            pred = 0
                            corrected = True
                            reason = f"All training WCs in group {group_id} DIED (mean=0.0)"
                # If they are an adult male, check if all WCs in the training group survived
                else:
                    if wc_mean == 1.0:
                        if ml_pred == 0:
                            pred = 1
                            corrected = True
                            reason = f"All training WCs in group {group_id} SURVIVED (mean=1.0)"
                            
        final_predictions.append(pred)
        
        if corrected:
            corrections_applied.append({
                'PassengerId': pid,
                'Name': name,
                'Sex': sex,
                'Pclass': pclass,
                'Ticket': ticket,
                'GroupId': group_id,
                'Original': ml_pred,
                'Corrected': pred,
                'Reason': reason
            })
            print(f"[{'MALE->1' if pred==1 else 'WC->0'}] {name} (Pclass {pclass}, Ticket {ticket})")
            print(f"  ML Pred: {ml_pred} -> Corrected: {pred}")
            print(f"  Reason: {reason}\n")
            
    print(f"Total corrections applied: {len(corrections_applied)}")
    
    # Save the updated submission file
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': final_predictions
    })
    submission.to_csv(output_sub_path, index=False)
    print(f"\nFinal corrected submission saved: {output_sub_path}")
    
    # Save a detailed log of the corrections
    if len(corrections_applied) > 0:
        corrections_df = pd.DataFrame(corrections_applied)
        log_path = "submissions/corrections_applied_log.csv"
        corrections_df.to_csv(log_path, index=False)
        print(f"Detailed corrections log saved: {log_path}")
        
    # Zip the final submission for Kaggle upload
    zip_path = output_sub_path.replace(".csv", ".zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(output_sub_path, os.path.basename(output_sub_path))
    print(f"Created zip archive for upload: {zip_path}")
    
    # Display final predictions distribution
    print("\n--- Final Submissions Distribution ---")
    print(submission['Survived'].value_counts(normalize=True).round(4))
    
    return zip_path


if __name__ == "__main__":
    apply_corrections(
        train_path="data/train.csv",
        test_path="data/test.csv",
        ml_sub_path="submissions/submission_ml_only_v3.csv",
        output_sub_path="submissions/submission_v3.csv"
    )
