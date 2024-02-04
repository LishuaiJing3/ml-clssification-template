import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import shap
import mlflow
import mlflow.sklearn
import joblib


# %%
def get_args():
    args = SimpleNamespace()
    args.log_level = 10
    args.log_directory = "logs"
    args.data_path = "data/sample_data.csv"
    args.model_name = "sample_churn"
    args.model_directory = "models/"

    return args


# Get the current working directory
current_working_directory = os.getcwd()
args = get_args()

# %%


def load_data(data_path):
    data = pd.read_csv(data_path)
    return data


def preprocess_data(data):
    # Check for missing values and handle them appropriately (if needed)

    # Encode categorical features using one-hot encoding
    categorical_features = [
        "Gender",
        "Location",
        "Account_Type",
        "Contract_Status",
    ]
    data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

    # Standardize or normalize numerical features to ensure comparable scales
    numerical_features = [
        "Age",
        "Tenure",
        "Income",
        "Spending_Score",
    ]
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    return data


def train_model(X_train, y_train):
    # Define hyperparameters to search
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # Choose a suitable machine learning algorithm for binary classification (Random Forest)
    model = xgb.XGBClassifier(random_state=42)

    # Initialize RandomizedSearchCV for hyperparameter tuning
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        random_state=42,
        n_jobs=-1,
    )

    # Perform hyperparameter tuning and train the model on the training set
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    return best_model


def main():
    """
    parser = argparse.ArgumentParser(description="Churn Prediction Training Script")
    parser.add_argument("--data_path", type=str, default="data/sample_data.csv", help="Path to the input data CSV file")
    parser.add_argument("--model_name", type=str, default="sample_churn", help="Name of the trained model")
    parser.add_argument("--model_directory", type=str, default="models/", help="Directory to save the trained model")
    args = parser.parse_args()
    """
    args = get_args()

    # Load data
    data = load_data(args.data_path)

    # Preprocess data
    data = preprocess_data(data)

    # Separate data into training and testing sets
    X = data.drop("Churn", axis=1)
    y = data["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(X_test.columns)
    print(y_test.columns)

    print(type(X_test))
    # Train model
    best_model = train_model(X_train, y_train)

    # Evaluate model
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print("Best Model - Accuracy:", accuracy)
    print("Best Model - Precision:", precision)
    print("Best Model - Recall:", recall)
    print("Best Model - F1 Score:", f1)

    # Initialize MLflow
    mlflow.start_run()

    # Log the best model in MLflow
    mlflow.sklearn.log_model(best_model, args.model_name)

    # Save the best model using joblib
    joblib.dump(best_model, f"{args.model_directory}/{args.model_name}.joblib")

    # Log the model artifact
    mlflow.log_artifact(
        f"{args.model_directory}/{args.model_name}.joblib", artifact_path="models"
    )

    # Log the best hyperparameters
    best_params = best_model.get_params()
    mlflow.log_params(best_params)

    # Use SHAP to explain the model predictions
    explainer = shap.Explainer(best_model, X_train.astype(float))
    shap_values = explainer.shap_values(X_test)

    # Example inference with the loaded model
    example_data = X_test.iloc[0:1]  # Replace with your own data
    prediction = best_model.predict(example_data)
    print("Predicted Churn (Best Model):", prediction)
    # Create a summary plot for a single customer's SHAP values
    shap.summary_plot(shap_values, X_test, show=False)
    # Use SHAP to explain the prediction for a specific customer
    customer_index = 0  # Change to the index of the customer you want to explain
    shap.plots.force(
        explainer.expected_value,
        shap_values[customer_index, :],
        X_test.iloc[customer_index, :],
        matplotlib=True,
    )
    shap.decision_plot(
        explainer.expected_value,
        shap_values[customer_index, :],
        X_test.iloc[customer_index, :],
    )
    # End MLflow run
    mlflow.end_run()


if __name__ == "__main__":
    main()