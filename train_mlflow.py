# ===== 1. Import the libraries =====
import pandas as pd  # Read and manipulate the CSV
import mlflow  # Main MLflow module
import mlflow.sklearn  # MLflow integration for scikit-learn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Four metrics to measure the quality of the model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ===== 2. Load the data (only once) =====
df = pd.read_csv("churn_data.csv")  # Table of 500 customers

y = df["churn"]  # Target: 1 = leaves, 0 = stays


# ===== 3. Function: prepare the data for a list of columns =====
def prepare(columns):
    # Convert the text column 'abonnement' into 0/1 columns (if present)
    X = pd.get_dummies(
        df[columns],
        drop_first=True
    )

    # Split: 80% training / 20% testing
    # random_state=42 ensures reproducible results
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


# ===== 4. Function: run ONE trial and record it in MLflow =====
def run_trial(name, n_estimators, max_depth, columns):
    X_train, X_test, y_train, y_test = prepare(columns)

    # Open a run named after the trial
    with mlflow.start_run(run_name=name):

        # Create the Random Forest model
        model = RandomForestClassifier(
            n_estimators=n_estimators,  # Number of trees
            max_depth=max_depth,  # Maximum depth of the trees
            random_state=42  # Seed for reproducibility
        )

        # Train the model
        model.fit(X_train, y_train)

        # Predict on the test customers
        pred = model.predict(X_test)

        # ===== Parameters: settings of this trial =====
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", ",".join(columns))

        # ===== Metrics: results of this trial =====
        mlflow.log_metric(
            "accuracy",
            accuracy_score(y_test, pred)
        )

        # zero_division=0 avoids warnings if no positive predictions are made
        mlflow.log_metric(
            "precision",
            precision_score(y_test, pred, zero_division=0)
        )

        mlflow.log_metric(
            "recall",
            recall_score(y_test, pred, zero_division=0)
        )

        mlflow.log_metric(
            "f1",
            f1_score(y_test, pred, zero_division=0)
        )

        # ===== Artifact: save the trained model =====
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle"
        )

        # Display the results
        print(
            name,
            "-> accuracy:",
            round(accuracy_score(y_test, pred), 3),
            "| f1:",
            round(f1_score(y_test, pred, zero_division=0), 3)
        )


# ===== 5. Main program =====

# Use the same experiment as in Step 1
mlflow.set_experiment("churn-prediction")

ALL = [
    "age",
    "salaire",
    "abonnement",
    "anciennete",
    "consommation"
]

# Trial 1: 100 trees, depth 5, all columns
run_trial("trial-1", 100, 5, ALL)

# Trial 2: simpler model (50 trees, depth 3)
run_trial("trial-2", 50, 3, ALL)

# Trial 3: without the salary column (salaire)
run_trial(
    "trial-3",
    100,
    5,
    ["age", "abonnement", "anciennete", "consommation"]
)