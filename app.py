import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Internal prediction microservice for customer churn inference",
    version="1.0.0"
)

def load_latest_model():
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Telecom_Churn_Prediction")
    if not experiment:
        raise RuntimeError("Experiment 'Telecom_Churn_Prediction' not found in MLflow.")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs:
        raise RuntimeError("No completed runs found for this experiment.")
        
    latest_run_id = runs[0].info.run_id
    model_uri = f"runs:/{latest_run_id}/model"
    print(f"Loading model from run ID: {latest_run_id}")
    return mlflow.sklearn.load_model(model_uri)

model = load_latest_model()

class CustomerFeatures(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "telecom-churn-api"}

@app.post("/predict")
def predict_churn(customer: CustomerFeatures):
    try:
        input_data = pd.DataFrame([customer.model_dump()])
        prediction = model.predict(input_data)[0]
        prediction_prob = model.predict_proba(input_data)[0][1]

        return {
            "churn_prediction": int(prediction),
            "churn_probability": round(float(prediction_prob), 4),
            "risk_level": "High" if prediction == 1 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))