from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("models/churn_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

class CustomerData(BaseModel):
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/model-info")
def model_info():
    return {
        "features": len(feature_columns)
    }

@app.get("/predict")
def predict():

    sample = pd.DataFrame(
        [[0] * len(feature_columns)],
        columns=feature_columns
    )

    prediction = int(model.predict(sample)[0])

    return {
        "prediction": prediction
    }

@app.post("/predict-customer")
def predict_customer(customer: CustomerData):

    sample = pd.DataFrame(
        [[0] * len(feature_columns)],
        columns=feature_columns
    )

    sample["SeniorCitizen"] = customer.SeniorCitizen
    sample["tenure"] = customer.tenure
    sample["MonthlyCharges"] = customer.MonthlyCharges
    sample["TotalCharges"] = customer.TotalCharges

    prediction = int(model.predict(sample)[0])

    probability = float(
        model.predict_proba(sample)[0][1]
    )

    return {
        "prediction": prediction,
        "churn_probability": probability
    }

@app.get("/feature-names")
def feature_names():
    return {
        "features": feature_columns[:10]
    }


#This is a commented line added to check the build pipeline.    