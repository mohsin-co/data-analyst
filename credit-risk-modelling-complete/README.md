# Credit Risk Modelling — Mohsin Iqbal

**Goal:** Predict default risk for consumer credit using tabular data. This repository contains a *complete, runnable* notebook with:
- Data cleaning & EDA
- Preprocessing with ColumnTransformer
- Baseline Logistic Regression
- RandomForestClassifier with class weights
- Metrics: ROC-AUC, PR-AUC, F1, Recall
- Feature importance
- Saved plots to `/assets/`

## Dataset
- Included synthetic sample: `data/german_credit_sample.csv` (structure similar to UCI German Credit; used for demo/offline runs).
- For a real dataset, replace `DATA_PATH` in the notebook with UCI/Kaggle credit data.

## How to run
1. Install requirements (Python 3.10+):
   ```bash
   pip install -r requirements.txt
   ```
2. Open the notebook:
   ```bash
   jupyter notebook credit_risk_model.ipynb
   ```

## Deliverables
- `credit_risk_model.ipynb` — full pipeline & results
- `/assets/` — ROC and PR curve images
- `results.json` — summary metrics (AUC, PR-AUC, F1, Recall)

## Notes
- Class imbalance is handled via `class_weight='balanced'`.
- Replace the sample dataset with a real dataset before showcasing to recruiters (the code path is documented in the notebook).
