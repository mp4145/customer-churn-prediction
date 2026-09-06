
# TELCO CUSTOMER CHURN PIPELINE: PROJECT SUMMARY           


1. DATA INGESTION & PIPELINE ARCHITECTURE (STREAMING ENGINE)
------------------------------------------------------------------------
- Transitioned successfully from simple local file downloads to a 
  highly scalable, industry-standard data streaming pipeline utilizing 
  Hugging Face's `IterableDataset`. 
- Engineered a "lazy transformation" framework. The pipeline operates 
  on-the-fly row-by-row, ensuring a zero-hard-drive-footprint approach
  that can scale up to terabytes of streaming data.
- Resolved strict data engineering constraints (VS Code output telemetry, 
  library type discrepancies, and multi-dimensional SHAP object maps) 
  using production-grade data adjustments.

2. PROGRAMMATIC DATA DIAGNOSTICS & FEATURE ENGINEERING
------------------------------------------------------------------------
- Single-Pass Audits caught 11 missing values in `TotalCharges` and 
  exposed a classic real-world database anomaly: all missing entries 
  belonged to brand-new accounts with exactly 0 months of `tenure`. 
- Used Welford's One-Pass Algorithm to dynamically map statistical 
  means and standard deviations directly over the live network stream.
- Leveraged Exploratory Data Analysis (EDA) to build domain-engineered 
  features that enhanced pattern exposure for our math matrices:
  - Account Segmentation: Grouped new, intermediate, and veteran risk states.
  - Product Stickiness: Aggregated multi-class ecosystem addon service ties.
  - Value-to-Investment Ratio: Calculated ongoing monthly strains relative 
    to overall contract histories.
  - Multi-Class & Binary Label Encodings: Automated numeric translations.

3. DYNAMIC SCALING & INITIAL ALGORITHM RACE
------------------------------------------------------------------------
- Executed a deterministic 3-set local distribution grid partition to 
  isolate pure Train, Validation, and final Test sets without leakage.
- Unified the data scaling workflow safely via Scikit-Learn's `StandardScaler`.
- Ran a performance race across core algorithms, showing tight baseline
  accuracies (~79-80%). Logistic Regression initially led in early testing 
  due to an exceptional feature engineering setup, yielding a validation 
  ROC-AUC score of 0.8388 and a churn recall of 0.53.

4. ADVANCED MODEL RUN & THE WINNER (BALANCED XGBOOST)
------------------------------------------------------------------------
- Evaluated LightGBM, SVM (RBF), and Balanced XGBoost. LightGBM reached 
  the highest pure validation accuracy (0.8012) but suffered from low 
  churn visibility (0.54 recall).
- Balanced XGBoost emerged as the clear operational winner. By adjusting 
  for the dataset's class imbalance using a dynamic sample ratio penalty 
  (scale_pos_weight = 2.69), it achieved a massive Churn Recall score 
  of 0.73 and a robust 0.8345 ROC-AUC score.
- <b>Business Justification</b>: In subscription telecom models, the massive 
  revenue saved by proactively capturing 73% of churned clients far 
  outweighs the negligible operational costs of minor false alarms.

========================================================================

## Run a local demo

The notebook is exploratory; the production-shaped wrapper lives in `predictor.py`. It keeps feature engineering in one place so the UI and API use the same prediction path.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The training command creates `artifacts/churn_model.joblib`. Do not commit that artifact if it contains sensitive training data or model metadata.

## Expose an API endpoint

Run the API separately:

```bash
uvicorn api:app --reload
```

Useful endpoints are `GET /health`, `POST /predict`, and `GET /docs` for the generated OpenAPI UI. For a real deployment, add authentication, privacy-aware logging, rate limiting, artifact versioning, and monitoring for input drift and prediction quality. Keep the decision threshold configurable and validate it against the business cost of false positives versus missed churn.

## Beginner notes: how the application works

### What is the notebook for?

`demo.ipynb` is the laboratory for this project. It loads the data, investigates missing values, creates features, compares algorithms, measures performance, and displays charts. Notebook variables exist only after cells have been executed in the correct order, so the notebook is useful for learning and experimentation but is not a reliable application server.

### Why are there separate Python files?

The application needs code that can be imported and run repeatedly without opening a notebook. Each file has one responsibility:

| File | Responsibility |
| --- | --- |
| `demo.ipynb` | Explore data and compare models |
| `predictor.py` | Convert one customer into model features and make a prediction |
| `train_model.py` | Train the chosen model and save it as an artifact |
| `app.py` | Provide a human-friendly Streamlit demo |
| `api.py` | Provide JSON endpoints for other software |
| `requirements.txt` | List the Python packages needed to run the project |

This is still one model, not several copies. The trained model and scaler are stored together in `artifacts/churn_model.joblib`. Both the Streamlit app and FastAPI load that same artifact and call the same code in `predictor.py`.

### What happens from input to prediction?

The important path is:

```text
Customer form or JSON
  |
  v
predictor.build_features()
  |
  v
23 engineered numeric features
  |
  v
Saved StandardScaler
  |
  v
Saved balanced XGBoost model
  |
  v
Churn probability and decision
```

The feature names, feature order, mappings, missing-value handling, and scaler are part of the model contract. Changing any of them requires retraining and creating a new artifact.

### How do I start from a clean environment?

Run these commands from the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The virtual environment keeps this project isolated from unrelated Python installations. On macOS and Linux, use `source .venv/bin/activate`; on Windows PowerShell, use `.venv\\Scripts\\Activate.ps1`.

### How is the model artifact created?

Run:

```bash
python train_model.py
```

This downloads the training stream, applies the shared feature builder, fits the scaler, trains balanced XGBoost, and writes `artifacts/churn_model.joblib`. Training can take longer than starting the apps because it processes the dataset and fits the model. You normally retrain only after changing the model or training data.

### How do I open the Streamlit demo?

Start it with:

```bash
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open that URL in a browser, fill in the customer fields, and select **Predict churn**. The form is for a person demonstrating the model. It is not an API endpoint and it does not retrain the model.

### How do I expose and test the API?

Start FastAPI with:

```bash
uvicorn api:app --reload
```

The server usually listens at `http://127.0.0.1:8000`.

Common endpoints:

- `GET /health`: confirms that the service is alive.
- `GET /docs`: opens an interactive Swagger page where you can try `/predict`.
- `POST /predict`: accepts customer JSON and returns a prediction.

Test the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

For a prediction, use the Swagger page or send JSON. A request must use `POST`, not a browser `GET`:

```bash
curl -X POST http://127.0.0.1:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0
  }'
```

The response contains `will_churn`, `churn_probability`, `threshold`, and `model_version`. A probability is not a certainty. It is the model's estimated risk based on the training data.

### Why does opening `http://127.0.0.1:8000/` show 404?

There is no root route in `api.py`. That is expected. Use `/health` or `/docs`. Similarly, opening `/predict` in a browser sends a `GET`, while the prediction route requires a `POST` body containing customer data.

### What if the API says the model artifact is missing?

Run the training command first:

```bash
python train_model.py
```

Confirm that `artifacts/churn_model.joblib` exists, then restart Uvicorn or Streamlit. The applications load the artifact when they start.

### What if I add a new model in the notebook?

Use the notebook to compare the new model fairly on the same validation and test data. If it is the operational winner, update the model construction in `train_model.py`, run `python train_model.py`, and restart the application. The UI and API can stay unchanged if the replacement exposes `predict_proba()` and uses the same feature matrix.

For a deep learning model, the saved format and prediction call may differ. In that case, update `predictor.py` so it loads the new format and converts its output into the same result fields. The API contract should remain stable even when the internal model changes.

### What if I change the features?

Update the feature builder and training code together, retrain the artifact, and test one known customer. Never train with one feature order and serve with another. The safest long-term arrangement is for the notebook to import the shared `build_features()` function instead of maintaining a second copy of the feature engineering logic.

### What belongs in production later?

This project is a local demo and learning scaffold. A production service would additionally need authentication, HTTPS, request limits, privacy-aware logging, automated tests, a model registry, artifact versioning, monitoring for feature drift, monitoring for model quality, and a documented process for retraining and rollback. It should also record why the churn threshold was chosen, because false positives and missed churn have different business costs.