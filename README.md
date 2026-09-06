
             TELCO CUSTOMER CHURN PIPELINE: PROJECT SUMMARY           


1. DATA INGESTION & PIPELINE ARCHITECTURE (STREAMING ENGINE)
------------------------------------------------------------------------
• Transitioned successfully from simple local file downloads to a 
  highly scalable, industry-standard data streaming pipeline utilizing 
  Hugging Face's `IterableDataset`. 
• Engineered a "lazy transformation" framework. The pipeline operates 
  on-the-fly row-by-row, ensuring a zero-hard-drive-footprint approach
  that can scale up to terabytes of streaming data.
• Resolved strict data engineering constraints (VS Code output telemetry, 
  library type discrepancies, and multi-dimensional SHAP object maps) 
  using production-grade data adjustments.

2. PROGRAMMATIC DATA DIAGNOSTICS & FEATURE ENGINEERING
------------------------------------------------------------------------
• Single-Pass Audits caught 11 missing values in `TotalCharges` and 
  exposed a classic real-world database anomaly: all missing entries 
  belonged to brand-new accounts with exactly 0 months of `tenure`. 
• Used Welford's One-Pass Algorithm to dynamically map statistical 
  means and standard deviations directly over the live network stream.
• Leveraged Exploratory Data Analysis (EDA) to build domain-engineered 
  features that enhanced pattern exposure for our math matrices:
  - Account Segmentation: Grouped new, intermediate, and veteran risk states.
  - Product Stickiness: Aggregated multi-class ecosystem addon service ties.
  - Value-to-Investment Ratio: Calculated ongoing monthly strains relative 
    to overall contract histories.
  - Multi-Class & Binary Label Encodings: Automated numeric translations.

3. DYNAMIC SCALING & INITIAL ALGORITHM RACE
------------------------------------------------------------------------
• Executed a deterministic 3-set local distribution grid partition to 
  isolate pure Train, Validation, and final Test sets without leakage.
• Unified the data scaling workflow safely via Scikit-Learn's `StandardScaler`.
• Ran a performance race across core algorithms, showing tight baseline
  accuracies (~79-80%). Logistic Regression initially led in early testing 
  due to an exceptional feature engineering setup, yielding a validation 
  ROC-AUC score of 0.8388 and a churn recall of 0.53.

4. ADVANCED MODEL RUN & THE WINNER (BALANCED XGBOOST)
------------------------------------------------------------------------
• Evaluated LightGBM, SVM (RBF), and Balanced XGBoost. LightGBM reached 
  the highest pure validation accuracy (0.8012) but suffered from low 
  churn visibility (0.54 recall).
• Balanced XGBoost emerged as the clear operational winner. By adjusting 
  for the dataset's class imbalance using a dynamic sample ratio penalty 
  (scale_pos_weight = 2.69), it achieved a massive Churn Recall score 
  of 0.73 and a robust 0.8345 ROC-AUC score.
• Business Justification: In subscription telecom models, the massive 
  revenue saved by proactively capturing 73% of churned clients far 
  outweighs the negligible operational costs of minor false alarms.

========================================================================