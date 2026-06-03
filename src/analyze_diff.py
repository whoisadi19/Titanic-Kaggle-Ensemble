"""
Titanic V4: Analysis of V2 predictions vs V3 predictions to understand exactly 
what happened, and build a better model from scratch.

Goal: understand which specific passengers V2 got right that V3 got wrong,
and vice versa, to inform a better strategy.
"""
import pandas as pd
import numpy as np

# Load all submissions
v2 = pd.read_csv("submissions/submission_v2.csv")
v3 = pd.read_csv("submissions/submission_v3.csv")
ml_v3 = pd.read_csv("submissions/submission_ml_only_v3.csv")
test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")

# V2 scored 0.79665 = 333 correct
# V3 scored 0.73923 = 309 correct 
# So V3 got 24 more predictions WRONG

# Compare V2 vs V3
merged = v2.merge(v3, on='PassengerId', suffixes=('_v2', '_v3'))
merged = merged.merge(ml_v3.rename(columns={'Survived': 'Survived_ml_v3'}), on='PassengerId')
merged = merged.merge(test, on='PassengerId')

diffs = merged[merged['Survived_v2'] != merged['Survived_v3']]

# Also compare V2 vs ML-only V3 (before corrections)
diffs_ml = merged[merged['Survived_v2'] != merged['Survived_ml_v3']]

with open("v2_vs_v3_analysis.txt", "w", encoding="utf-8") as f:
    f.write("=== V2 vs V3 Comparison ===\n\n")
    f.write(f"V2 score: 0.79665 (333 correct)\n")
    f.write(f"V3 score: 0.73923 (309 correct)\n")
    f.write(f"V3 got 24 MORE passengers wrong than V2\n\n")
    
    f.write(f"Total predictions that differ between V2 and V3: {len(diffs)}\n")
    f.write(f"Total predictions that differ between V2 and ML-only V3: {len(diffs_ml)}\n\n")
    
    # How many differ between V2 and ML-only V3?
    f.write("=== V2 vs ML-only V3 (before corrections) ===\n")
    f.write(f"Predictions that differ: {len(diffs_ml)}\n")
    v2_0_mlv3_1 = diffs_ml[diffs_ml['Survived_v2'] < diffs_ml['Survived_ml_v3']]
    v2_1_mlv3_0 = diffs_ml[diffs_ml['Survived_v2'] > diffs_ml['Survived_ml_v3']]
    f.write(f"  V2=0, ML_V3=1: {len(v2_0_mlv3_1)}\n")
    f.write(f"  V2=1, ML_V3=0: {len(v2_1_mlv3_0)}\n\n")
    
    f.write("=== V2 vs V3 (after corrections) ===\n")
    f.write(f"Predictions that differ: {len(diffs)}\n")
    v2_0_v3_1 = diffs[diffs['Survived_v2'] < diffs['Survived_v3']]
    v2_1_v3_0 = diffs[diffs['Survived_v2'] > diffs['Survived_v3']]
    f.write(f"  V2=0, V3=1: {len(v2_0_v3_1)} (V3 flipped to survive)\n")
    f.write(f"  V2=1, V3=0: {len(v2_1_v3_0)} (V3 flipped to die)\n\n")
    
    f.write("--- Passengers V3 flipped to SURVIVE (V2=0 -> V3=1) ---\n")
    for _, row in v2_0_v3_1.iterrows():
        f.write(f"  PID {row['PassengerId']}: {row['Name']} ({row['Sex']}, Pclass {row['Pclass']}, Ticket {row['Ticket']})\n")
        f.write(f"    ML_V3={row['Survived_ml_v3']}, V3_corrected={row['Survived_v3']}\n")
    
    f.write("\n--- Passengers V3 flipped to DIE (V2=1 -> V3=0) ---\n")
    for _, row in v2_1_v3_0.iterrows():
        f.write(f"  PID {row['PassengerId']}: {row['Name']} ({row['Sex']}, Pclass {row['Pclass']}, Ticket {row['Ticket']})\n")
        f.write(f"    ML_V3={row['Survived_ml_v3']}, V3_corrected={row['Survived_v3']}\n")

    # Let's also check the ML-only V3 vs V2 — the ML model ITSELF differs
    f.write("\n\n=== ML-only V3 vs V2 (differences from the model, before corrections) ===\n")
    for _, row in diffs_ml.iterrows():
        f.write(f"  PID {row['PassengerId']}: {row['Name']} ({row['Sex']}, Pclass {row['Pclass']})\n")
        f.write(f"    V2={row['Survived_v2']}, ML_V3={row['Survived_ml_v3']}\n")

print("Analysis saved to v2_vs_v3_analysis.txt")
