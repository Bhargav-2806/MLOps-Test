# Stage 1: Pipeline Initialization & Model Training

## The Machine Learning Software & Dataset
We are building a **Telecom Customer Churn Predictor**, an internal backend service that evaluates customer profiles and predicts whether they are likely to cancel their subscription. 
*   **The Dataset:** We utilized the open-source IBM `WA_Fn-UseC_-Telco-Customer-Churn.csv`. It contains 7,043 rows of historical customer data, detailing features like monthly charges, contract lengths, and a final `Churn` column indicating if they left.
*   **The Objective:** The script uses historical patterns to learn *why* people leave, creating a mathematical rulebook (the model) that can score new customers.

## Core Libraries Used
*   **Pandas:** The data manipulator. It loads the raw CSV into a structured table (dataframe) so we can drop useless columns and fix missing numbers.
*   **Scikit-Learn (sklearn):** The machine learning engine. It splits the data, converts text categories to numbers, and trains the actual predictive algorithm.
*   **MLflow:** The experiment tracker. It silently observes the training process, permanently logging our settings, scoring the accuracy, and saving the final model artifact securely.
*   **DVC:** Data Version Control. It creates tiny pointer files (`.dvc`) so Git can track large datasets without uploading massive CSVs to the repository.

## Demystifying the Training Code & Output
The `train.py` script executes a standard machine learning workflow without manual intervention:
1.  **Loading & Cleaning:** It ingests the CSV and removes irrelevant data like `customerID`.
2.  **Encoding:** Algorithms only understand math. The script scans for text columns (like 'Female' or 'Month-to-Month') and converts them into numeric identifiers so the model can process them.
3.  **Training & Evaluation:** It splits the data, hiding 20% to test the model later. It trains a Random Forest algorithm on the 80%, then scores its predictive performance on the hidden 20%.
4.  **The Output:** The script achieved an **Accuracy of 80.13%**. It logged this metric to MLflow, along with a security fix (`skops_trusted_types`) that ensures the saved model structures are trusted and safe from malicious code execution.

```text
[Raw Data: CSV] ---> [DVC Tracks Data] ---> [train.py cleans & trains]
                                                   |
                                                   v
                                [MLflow Logs Metrics & Saves Model Artifact]