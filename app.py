# ===== 1. Import the libraries =====
import os  # Read the environment variables
from typing import Literal  # Limit the possible values of a field

import mlflow  # Load the model (Registry or folder)
import mlflow.sklearn
import pandas as pd  # Build the table expected by the model

from fastapi import FastAPI  # The framework of the API
from pydantic import BaseModel  # Describe and validate the received data


# ===== 2. Create the application =====
app = FastAPI(title="Churn API")  # The title is displayed in the /docs page


# ===== 3. Read the configuration from the environment variables =====
# Nothing is written "hard-coded": the same code works on the computer, in a
# single container, or in Docker Compose. Only the variables change.

# os.getenv("NAME", "value"): value of the variable, or the default value if it does not exist
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = os.getenv("MODEL_URI", "models:/churn-model@champion")


# ===== 4. Load the champion ONLY ONCE, at startup =====
# MODEL_URI can be a Registry address (models:/...) or a simple folder (model)

mlflow.set_tracking_uri(TRACKING_URI)

model = mlflow.sklearn.load_model(MODEL_URI)

print(f"Model loaded from: {MODEL_URI}")


# ===== 5. Describe the expected data (validated automatically) =====
class Customer(BaseModel):
    age: int  # Whole number
    salaire: float  # Decimal number (salary)
    abonnement: Literal["Basique", "Standard", "Premium"]  # 3 possible values
    anciennete: float  # Seniority / tenure
    consommation: float  # Usage


# ===== 6. Route /health: check that the API is alive =====
@app.get("/health")
def health():
    return {"status": "ok"}


# ===== 7. Route /predict: receive a customer, return the prediction =====
@app.post("/predict")
def predict(customer: Customer):

    # a) Turn the received customer into a one-row table
    row = pd.DataFrame([customer.model_dump()])

    # b) Turn the text "abonnement" into 0/1 columns, as during training
    row = pd.get_dummies(row)

    # c) Keep the model's columns, in the right order; missing ones are 0
    row = row.reindex(columns=model.feature_names_in_, fill_value=0)

    # d) Predict: 1 = leaves, 0 = stays
    result = int(model.predict(row)[0])

    return {
        "churn": result,
        "message": "The customer will leave" if result == 1 else "The customer stays"
    }