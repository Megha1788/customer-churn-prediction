# Customer Churn Prediction

Complete machine learning assignment project for predicting customer churn using the IBM Telco Customer Churn dataset.

The project covers the complete machine learning lifecycle:

**Data Understanding → EDA → Feature Engineering → Leakage-Safe Preprocessing → Model Training → Hyperparameter Tuning → Evaluation → Model Saving → REST API**

---

## 1. Problem Statement

Customer churn is an important business problem because identifying customers who are likely to leave can allow a business to take proactive retention actions.

The objective of this project is to build a machine learning solution that predicts whether a customer is likely to churn based on customer demographics, services, contract information, tenure and billing characteristics.

The IBM Telco Customer Churn dataset contains:

- 7,043 customers
- 21 columns/features
- Binary target variable: `Churn`
- Approximately 26.5% of customers in the dataset churned

Because the dataset is imbalanced, accuracy alone is not sufficient for evaluating the model. Precision, recall and F1 score are also considered.

---

## 2. Assignment Requirements

The project covers the following required work:

- Data loading and understanding
- Data type and structure analysis
- Missing-value analysis
- Duplicate analysis
- Numerical and categorical feature identification
- Target variable analysis
- Data cleaning and preprocessing
- 70:30 stratified train-test split
- `random_state=42`
- At least 5 meaningful EDA visualizations
- Business interpretation of EDA results
- Two engineered features
- At least two Decision Tree configurations
- Model comparison
- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- Feature importance
- Decision Tree visualization
- Saved preprocessing/model pipeline
- FastAPI REST API
- `POST /predict` endpoint
- JSON input
- Churn prediction and probability
- Invalid-input handling

---

## 3. Bonus Work

The project also includes:

1. **Hyperparameter tuning** using `GridSearchCV`
2. **Class imbalance handling** using `class_weight="balanced"`
3. **Additional model comparison** using Random Forest
4. Additional business segmentation during EDA

---

## 4. Dataset

The project uses the IBM Telco Customer Churn dataset.

Important columns include:

### Customer information
- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`

### Service information
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`

### Account information
- `tenure`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`

### Billing information
- `MonthlyCharges`
- `TotalCharges`

### Target
- `Churn`

`customerID` is treated as an identifier and is not used as a predictive feature.

---

## 5. Data Understanding and Preparation

The notebook performs:

- Dataset shape and structure inspection
- Data type inspection
- Descriptive statistics
- Missing-value analysis
- Duplicate analysis
- Target distribution analysis
- Numerical/categorical feature identification

There are 11 blank values in `TotalCharges`. These values are handled as part of the preprocessing pipeline.

No duplicate customer records were identified.

---

## 6. Exploratory Data Analysis

The notebook includes six visualizations:

1. Overall churn distribution
2. Churn by contract type
3. Churn by internet service
4. Churn by payment method
5. Churn by tenure
6. Monthly charges by churn status

The EDA shows several customer segments with substantially different observed churn rates, including differences by contract type, tenure, internet service and payment method.

These observations describe associations observed in the dataset and should not be interpreted as proof of causation.

---

## 7. Train-Test Split and Data Leakage Prevention

The dataset is split into training and testing data before preprocessing.

```python
train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)
```

To avoid data leakage, preprocessing is fitted only on the training data.

The complete saved pipeline follows:

```text
Feature Engineering
        ↓
Imputation
        ↓
One-Hot Encoding
        ↓
Decision Tree
```

The categorical encoder uses:

```python
OneHotEncoder(handle_unknown="ignore")
```

This allows unseen categorical values to be handled safely.

The final `churn_model.pkl` contains the complete pipeline rather than only the classifier.

---

## 8. Feature Engineering

Two business-oriented features are created.

### AvgMonthlySpend

Calculated from:

```text
TotalCharges / tenure
```

with protection for zero tenure.

### NumOptionalServices

Counts optional services including:

- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies

These features provide additional representations of customer spending and service adoption.

---

## 9. Preprocessing Pipeline

Numerical features use median imputation.

Categorical features use most-frequent imputation followed by one-hot encoding.

The final estimator is a Decision Tree Classifier.

The complete preprocessing and model are stored together in the saved pipeline.

---

## 10. Decision Tree Models

Multiple Decision Tree configurations were evaluated to understand the effect of tree complexity and class imbalance handling.

The final model is selected using test-set performance and F1 score.

---

## 11. Hyperparameter Tuning

GridSearchCV is used to tune the Decision Tree.

The current notebook uses:

```text
3-fold cross-validation
Scoring = F1
```

The tuning considers:

- `max_depth`
- `min_samples_split`
- `class_weight`

Selected configuration:

```text
max_depth = 4
min_samples_split = 2
class_weight = balanced
```

---

## 12. Model Comparison

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Decision Tree A | 79.37% | 61.38% | 60.07% | 60.72% |
| Decision Tree B | 75.82% | 54.88% | 50.09% | 52.38% |
| Balanced Decision Tree | 69.76% | 46.15% | 83.42% | 59.43% |
| Tuned Decision Tree | **74.87%** | **51.83%** | **75.76%** | **61.55%** |
| Random Forest | 78.23% | 61.72% | 47.42% | 53.63% |

The tuned Decision Tree achieved the highest F1 score among the Decision Tree configurations evaluated in this project.

---

## 13. Final Model

The final model is a tuned Decision Tree Classifier.

```text
Algorithm: Decision Tree Classifier
max_depth: 4
min_samples_split: 2
class_weight: balanced
Cross-validation: 3-fold
Scoring: F1
```

### Final Test Results

| Metric | Result |
|---|---:|
| Accuracy | 74.87% |
| Precision | 51.83% |
| Recall | 75.76% |
| F1 Score | 61.55% |

---

## 14. Evaluation

The final model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

Recall is particularly relevant for a churn-retention use case because identifying more actual churners can provide more opportunities for potential retention actions.

The final model achieved approximately 75.8% recall, identifying approximately three quarters of the actual churners in the test set.

Precision was approximately 51.8%. There is a business trade-off between recall and precision: increasing recall can identify more potential churners but may also increase false positives.

---

## 15. Feature Importance and Model Interpretation

Decision Trees provide an interpretable model structure.

The project analyzes feature importance and generates a Decision Tree visualization.

The tree shows how customer records are recursively separated based on feature values to produce different churn predictions.

This provides additional transparency compared with treating the model as a complete black box.

---

## 16. Model Saving

The complete trained pipeline is saved using `joblib`.

```text
model/churn_model.pkl
```

The saved object includes:

```text
Feature Engineering
        ↓
Imputation
        ↓
One-Hot Encoding
        ↓
Decision Tree
```

This allows the same preprocessing and model logic to be reused for new customers.

---

## 17. REST API

The project exposes the saved model through a FastAPI REST API.

Endpoint:

```text
POST /predict
```

The endpoint accepts customer information as JSON and returns:

- Predicted churn class
- Churn probability

---

## 18. Running the Project

### Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Jupyter Notebook

Open:

```text
notebook/churn_analysis.ipynb
```

Run the notebook from the beginning.

The notebook performs data understanding, preparation, EDA, feature engineering, model training, tuning, evaluation, feature importance and model saving.

The final pipeline is saved to:

```text
model/churn_model.pkl
```

---

## 19. Start the FastAPI Application

From the project root:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 5000
```

The API runs at:

```text
http://127.0.0.1:5000
```

---

## 20. Swagger API Documentation

Open:

```text
http://127.0.0.1:5000/docs
```

FastAPI provides an interactive Swagger interface for testing `/predict`.

---

## 21. API Endpoint

### POST `/predict`

Example request:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
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
  "MonthlyCharges": 85.50,
  "TotalCharges": "427.50"
}
```

Example response:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.78
}
```

The actual probability is generated by the saved model and may differ depending on the model version and input.

A ready-to-use request is available in:

```text
sample_request.json
```

---

## 22. Input Validation and Error Handling

The API validates required customer fields.

If required fields are missing, the API returns HTTP 400 and identifies the missing fields.

Example:

```json
{
  "error": "Missing required fields",
  "fields": [
    "tenure",
    "Contract"
  ]
}
```

Invalid input that cannot be processed by the pipeline is also handled with an HTTP 400 response.

---

## 23. Project Structure

```text
customer_churn_project/
│
├── data/
│   ├── TelcoCustomerChurn.csv
│   └── TelcoCustomerChurn - Data Dictionary.csv
│
├── notebook/
│   └── churn_analysis.ipynb
│
├── model/
│   └── churn_model.pkl
│
├── src/
│   └── churn_pipeline.py
│
├── app.py
├── sample_request.json
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 24. End-to-End Workflow

```text
IBM Telco Dataset
        ↓
Data Understanding
        ↓
Data Quality Checks
        ↓
70:30 Stratified Train/Test Split
        ↓
Feature Engineering
        ↓
Preprocessing Pipeline
        ↓
Decision Tree Models
        ↓
GridSearchCV Tuning
        ↓
Model Evaluation
        ↓
Feature Importance
        ↓
Save Complete Pipeline
        ↓
FastAPI REST API
        ↓
POST /predict
        ↓
Churn Prediction + Probability
```

---

## 25. Business Value

The solution can support a proactive customer-retention workflow by identifying customers who may be at higher risk of churn.

Potential uses include:

- Proactive retention campaigns
- Customer risk segmentation
- Prioritization of retention efforts
- Identification of customer groups with higher observed churn
- Explainable model-based decision support

The model should be used as a decision-support tool rather than as the sole basis for customer decisions.

---

## 26. Data Leakage Prevention Statement

> "To avoid data leakage, I split the data before fitting preprocessing. The complete preprocessing and model are saved in one pipeline, so the same transformations are reused for the test set and for new unseen customers through the API."

---

## 27. Demo

Recommended demonstration sequence:

1. Open the Jupyter notebook.
2. Show dataset structure and data-quality checks.
3. Show EDA visualizations and business observations.
4. Show feature engineering.
5. Show the preprocessing pipeline.
6. Show Decision Tree model comparison.
7. Show final evaluation metrics.
8. Show feature importance and Decision Tree visualization.
9. Show the saved `churn_model.pkl`.
10. Start the FastAPI application.
11. Open Swagger at `/docs`.
12. Execute `POST /predict`.
13. Show churn prediction and probability.
14. Demonstrate invalid-input handling.

---

## 28. Conclusion

This project demonstrates an end-to-end customer churn prediction solution covering:

**Data Understanding → EDA → Feature Engineering → Leakage Prevention → Preprocessing → Model Training → Hyperparameter Tuning → Evaluation → Interpretation → Model Saving → REST API**

The final solution provides a reusable machine learning pipeline that can process new customer records and return a churn prediction and probability through a FastAPI endpoint.
