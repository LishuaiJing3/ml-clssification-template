#%%
import os
import pandas as pd
import numpy as np
import shap #type: ignore
import joblib #tupe: ignore
from typing import Tuple
import argparse


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters:
        data_path (str): The file path to the CSV data.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    data = pd.read_csv(data_path)
    return data

def load_artifacts(args: argparse.Namespace) -> Tuple[joblib, joblib]:
    """
    Load the trained model and preprocessor artifacts.

    Parameters:
        args (argparse.Namespace): Arguments containing model and preprocessor paths.

    Returns:
        Tuple: A tuple containing the loaded model and preprocessor.
    """
    model_path = os.path.join(args.model_directory, args.model_name)
    preprocessor_path = os.path.join(args.model_directory, args.preprocessor_name)
    
    loaded_model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    return loaded_model, preprocessor

def preprocess_data(preprocessor: joblib, data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to the data using the loaded preprocessor.

    Parameters:
        preprocessor (joblib): The preprocessor loaded from a joblib file.
        data (pd.DataFrame): The data to be preprocessed.

    Returns:
        pd.DataFrame: Preprocessed data as a DataFrame.
    """
    data_processed = preprocessor.transform(data)
    feature_names = preprocessor.get_feature_names_out()
    data_df = pd.DataFrame(data_processed, columns=feature_names)
    
    return data_df

def explain_single_prediction(model: joblib, explainer: shap.Explainer, data_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use the loaded model to make a prediction and explain it using SHAP for a single record.

    Parameters:
        model (joblib): The loaded predictive model.
        explainer (shap.Explainer): The SHAP explainer object initialized with the model.
        data_df (pd.DataFrame): The preprocessed data for making the prediction.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the prediction and SHAP explanation.
    """
    prediction = model.predict(data_df)
    shap_explanation = explainer.shap_values(data_df)

    return prediction, shap_explanation

def main():
    """
    Main function to execute the inference process, including data loading, preprocessing,
    model prediction, and SHAP explanation for either a single record or a batch of data.
    """
    class Args:
        model_directory = "models"
        model_name = "sample_churn.joblib"
        preprocessor_name = "preprocessor.joblib"
        data_path = "data/inference_data.csv"
        single_record = True  # Set to True for single record inference

    args = Args()

    # Load the trained model and preprocessor
    model, preprocessor = load_artifacts(args)

    if args.single_record:
        # Example single record for testing
        example_record = {
            'Age': 31,
            'Gender': 'Male',  # Will be one-hot encoded during preprocessing
            'Location': 'City',  # Will be one-hot encoded during preprocessing
            'Account_Type': 'Checking',  # Will be one-hot encoded during preprocessing
            'Tenure': 7,
            'Contract_Status': '2-Year',  # Will be one-hot encoded during preprocessing
            'Income': 67101,
            'Spending_Score': 5.6,
            # 'Churn': 0  # Excluded for prediction
        }
        # Convert the dictionary to DataFrame
        single_record_df = pd.DataFrame([example_record])

        # Preprocess the single record
        data_preprocessed = preprocess_data(preprocessor, single_record_df)

        # Initialize SHAP explainer
        explainer = shap.Explainer(model)

        # Make prediction and explain for the single record
        prediction, shap_explanation = explain_single_prediction(model, explainer, data_preprocessed)
        print(f"Predicted Churn for Single Record: {prediction}")
        shap.force_plot(explainer.expected_value, shap_explanation, data_preprocessed, matplotlib=True)
    else:
        # Load the data for batch inference
        data = load_data(args.data_path)

        # Preprocess the data
        data_preprocessed = preprocess_data(preprocessor, data)

        # Initialize SHAP explainer
        explainer = shap.Explainer(model)

        # Make predictions and explain for batch inference
        predictions = model.predict(data_preprocessed)
        shap_values = explainer.shap_values(data_preprocessed)

        for i in range(min(len(data_preprocessed), 5)):
            print(f"Customer {i+1} - Predicted Churn: {predictions[i]}")
            shap.force_plot(explainer.expected_value, shap_values[i,:], data_preprocessed.iloc[i,:], matplotlib=True)

if __name__ == "__main__":
    main()


# %%
