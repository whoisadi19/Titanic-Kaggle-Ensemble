import pandas as pd

v2 = pd.read_csv("submissions/submission_v2.csv")
v4 = pd.read_csv("submissions/submission_v4.csv")
test = pd.read_csv("data/test.csv")

merged = v2.rename(columns={'Survived': 'V2_Pred'}).merge(
    v4.rename(columns={'Survived': 'V4_Pred'}), on='PassengerId'
)
merged = merged.merge(test, on='PassengerId')

diffs = merged[merged['V2_Pred'] != merged['V4_Pred']]

with open("v2_vs_v4_diff.txt", "w", encoding="utf-8") as f:
    f.write(f"Total differences: {len(diffs)}\n\n")
    for _, row in diffs.iterrows():
        f.write(f"PID {row['PassengerId']}: {row['Name']} (Sex: {row['Sex']}, Age: {row['Age']}, Pclass: {row['Pclass']}, Ticket: {row['Ticket']}, SibSp: {row['SibSp']}, Parch: {row['Parch']})\n")
        f.write(f"  V2 Predicted: {row['V2_Pred']}\n")
        f.write(f"  V4 Predicted: {row['V4_Pred']}\n\n")

print("Saved to v2_vs_v4_diff.txt")
