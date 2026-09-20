
from pathlib import Path
import sys
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from churn_pipeline import ChurnFeatureEngineer  # required for joblib deserialization

model = joblib.load(ROOT / "model" / "churn_model.pkl")

app = FastAPI(title="Customer Churn Prediction API", version="1.0")

REQUIRED_FIELDS = [
    "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
    "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"
]

@app.get("/")
def root():
    return {"message":"Customer Churn Prediction API","predict":"POST /predict","docs":"/docs"}

@app.post("/predict")
def predict(payload: dict):
    missing = [x for x in REQUIRED_FIELDS if x not in payload]
    if missing:
        return JSONResponse(status_code=400, content={"error":"Missing required fields","fields":missing})
    try:
        row = {k:v for k,v in payload.items() if k != "customerID"}
        X_new = pd.DataFrame([row])
        p = int(model.predict(X_new)[0])
        probability = float(model.predict_proba(X_new)[0,1])
        return {"prediction":"Yes" if p else "No","churn_probability":round(probability,4)}
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error":"Invalid input","details":str(exc)})
