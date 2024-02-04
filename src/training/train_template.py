#%%
from typing import List, cast, Tuple
import pandas as pd #type: ignore
from sklearn.model_selection import train_test_split, RandomizedSearchCV #type: ignore
from sklearn.model_selection import RandomizedSearchCV #type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score #type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder #type: ignore
from sklearn.compose import ColumnTransformer #type: ignore
from sklearn.pipeline import Pipeline #type: ignore
from sklearn.impute import SimpleImputer #type: ignore
from xgboost import XGBClassifier
import mlflow #type: ignore
import shap #type: ignore
from types import SimpleNamespace
from sklearn.base import BaseEstimator # type: ignore
import joblib #type: ignore
from sklearn.base import BaseEstimator  # type: ignore
#%%
def get_args() -> SimpleNamespace:
    """
    Creates a namespace with configuration arguments.

    Returns:
        SimpleNamespace: Configuration arguments.
    """
    args = SimpleNamespace()
    args.log_level = 10
    args.log_directory = "logs"
    args.data_path = "data/sample_data.csv"
    args.model_name = "sample_churn"
    args.model_directory = "models/"
    return args

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from a specified path.

    Parameters:
        data_path (str): The path to the CSV file containing the data.

    Returns:
        pd.DataFrame: The loaded data.
    """
    data = pd.read_csv(data_path)
    return data

def preprocess_data(args: SimpleNamespace, data: pd.DataFrame, categorical_features: list, numerical_features: list, fit_transform: bool=True) -> Tuple[pd.DataFrame, list]:
    """
    Preprocesses the data by applying one-hot encoding to categorical features and standard scaling to numerical features.

    Parameters:
        args (SimpleNamespace): Configuration arguments including model directory for saving or loading the preprocessor.
        data (pd.DataFrame): The data to preprocess.
        categorical_features (list): List of names of the categorical features.
        numerical_features (list): List of names of the numerical features.
        fit_transform (bool, optional): Whether to fit the preprocessor or just transform the data. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, list]: A tuple containing the preprocessed data as a DataFrame and a list of feature names after preprocessing.
    """
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='drop')
    
    if fit_transform:
        data_processed = preprocessor.fit_transform(data)
        joblib.dump(preprocessor, f'{args.model_directory}/preprocessor.joblib')
    else:
        preprocessor = joblib.load(f'{args.model_directory}/preprocessor.joblib')
        data_processed = preprocessor.transform(data)
    
    feature_names = get_feature_names(preprocessor)
    
    data_df = pd.DataFrame(data_processed, columns=feature_names)
    
    return data_df, feature_names

def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts feature names from a ColumnTransformer object.

    Parameters:
        preprocessor (ColumnTransformer): The preprocessor from which to extract feature names.

    Returns:
        List[str]: A list of feature names.
    """
    #feature_names = preprocessor.get_feature_names_out()
    feature_names = cast(List[str], preprocessor.get_feature_names_out())

    return feature_names

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
    """
    Trains a model using RandomizedSearchCV for hyperparameter tuning.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        BaseEstimator: The trained model.
    """
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    model = XGBClassifier(random_state=42)
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    return best_model


def main()->None:
    args = get_args()

    # Load data
    data = load_data(args.data_path)

    # Separate features and target before preprocessing
    X = data.drop("Churn", axis=1)  # Features
    y = data["Churn"]  # Target variable

    categorical_features = ["Gender", "Location", "Account_Type", "Contract_Status"]
    numerical_features = ["Age", "Tenure", "Income", "Spending_Score"]
    
    # Preprocess features only, assuming preprocess_data doesn't expect 'Churn' column
    X_preprocessed, feature_names = preprocess_data(args, X, categorical_features, numerical_features, fit_transform=True)

    # Split processed features and target into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

    # Continue with training, evaluation, and explanation as before
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
#%%
if __name__ == "__main__":
    main()
# %%
