import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

print("1. Loading dataset...")
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("2. Preprocessing data...")
df = df.drop('customerID', axis=1) # Drop ID as it holds no predictive value
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Robust encoding for all non-numeric columns
for col in df.select_dtypes(include=['object', 'string', 'category']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("3. Training Random Forest model...")
mlflow.set_experiment("Telecom_Churn_Prediction")

with mlflow.start_run():
    n_estimators = 100
    max_depth = 5

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    print("4. Evaluating model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model trained successfully. Accuracy: {accuracy:.4f}")

    print("5. Logging to MLflow...")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    
    signature = infer_signature(X_train, predictions)
    mlflow.sklearn.log_model(
     sk_model=model, 
     artifact_path="model", 
     signature=signature,
     skops_trusted_types=["sklearn.tree._tree.Tree"]
 )

print("Pipeline complete!")