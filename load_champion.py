import mlflow
import mlflow.sklearn # to reload a scikit-learn model
import pandas as pd
# Same database as Lab 3
mlflow.set_tracking_uri("sqlite:///mlflow.db")
# 1. Load the champion BY ITS ALIAS: we know neither the version nor the run
model = mlflow.sklearn.load_model("models:/churn-model@champion")
print("Model loaded :", type(model).__name__)
print("Expected variables :", list(model.feature_names_in_))
# 2. A new customer (values of your choice). The subscription columns are
# already converted to 0/1: here, Standard subscription
customer = pd.DataFrame([{
"age": 35, "salaire": 2500, "anciennete": 1, "consommation": 10,
"abonnement_Premium": False, "abonnement_Standard": True,
}])
# 3. Keep only the columns known to the model, in the right order
customer = customer[list(model.feature_names_in_)]
# 4. Predict: 1 = the customer will leave, 0 = stays
print("Will the customer leave? ->", "YES" if model.predict(customer)[0] == 1 else "NO")