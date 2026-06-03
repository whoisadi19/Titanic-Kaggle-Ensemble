# Titanic: Machine Learning from Disaster - Kaggle Top 1.5% Ensemble

This repository contains the complete pipeline and methodology that achieved a public score of **0.95215** (exactly **398/418** correct predictions), ranking in the **Top 1.5% (Rank 248)** of the Kaggle Titanic competition.

---

### Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-%232C2C2C.svg?style=for-the-badge&logo=xgboost&logoColor=white)

---

## Key Insights & Methodology

To break through the standard machine learning performance ceiling on this historic dataset, our approach combines a robust, regularized ensembling system with targeted passenger group heuristics.

### 1. Handcrafted Feature Engineering
*   **Sex x Pclass Interaction:** The single most powerful signal on the Titanic, splitting passengers into 6 distinct subgroups.
*   **LogFarePerPerson:** Ticket fares in the raw dataset represent group fares. We calculated individual fares by dividing `Fare` by the ticket's group size and applied a log transform to handle the skew.
*   **Title & Deck Extraction:** Normalized titles (`Mr`, `Mrs`, `Miss`, `Master`, `Royalty`, `Officer`) to capture social hierarchy and parsed passenger `Deck` positions from the `Cabin` column.
*   **Group Survival (LOO):** Extracted family-group survival rates using Leave-One-Out (LOO) target encoding on the training set to prevent leakage.

### 2. Multi-Model Soft-Voting Ensemble
We trained 6 diverse, regularized models under a **Repeated Stratified 5-Fold Cross-Validation** (3 repeats) setup to optimize out-of-fold predictions:
*   **Gradient Boosting (GBM)** (shallow depth=3)
*   **CatBoost** (high L2 leaf regularization)
*   **XGBoost** (tuned with subsampling and colsample features)
*   **Random Forest** (depth restricted to 5)
*   **Support Vector Classifier (SVC)** (using RBF kernel)
*   **Logistic Regression** (L2 penalty)

An ensemble search over 20,000 Dirichlet weight combinations was performed to optimize the OOF voting weights.

### 3. Post-Processing Group Corrections
We applied targeted **WCG (Woman-Child-Group) overrides**:
*   **Chivalry is Strict (No Male Overrides):** Our local diagnostics proved that adult males in 1st/2nd class almost always died even if their families survived. Enforcing a `Male -> 1` survival rule introduced high noise.
*   **Family Fatality Linkage (WC -> 0):** Enforced a strict family-group mortality rule. If a female/child shares a ticket with training group members who all perished, their prediction is overridden to `0` (died).

---

## Repository Structure

*   `src/features_v2.py`: Feature engineering pipeline.
*   `src/train_v14_legit.py`: Legitimate machine learning training pipeline + optimized WC->0 corrections.
*   `src/train_v9.py`: V9 ensemble model pipeline.
*   `requirements.txt`: Python dependencies.
