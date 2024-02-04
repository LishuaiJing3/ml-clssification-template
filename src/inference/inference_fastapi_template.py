from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd #type: ignore
import joblib #type: ignore

# Define your Pydantic models for data validation
class Record(BaseModel):
    Age: float
    Gender: str
    Location: str
    Account_Type: str
    Tenure: float
    Contract_Status: str
    Income: float
    Spending_Score: float

class Records(BaseModel):
    records: List[Record]

app = FastAPI()

# Load the trained model and preprocessor
model = joblib.load("../../models/sample_churn.joblib")
preprocessor = joblib.load("../../models/preprocessor.joblib")

@app.post("/predict/")
async def make_prediction(record: Record):
    input_data = pd.DataFrame([record.dict()])
    data_preprocessed = preprocess_data(preprocessor, input_data)
    prediction = model.predict(data_preprocessed)
    return {"prediction": prediction.tolist()}

@app.post("/predict_batch/")
async def make_batch_prediction(records: Records):
    input_data = pd.DataFrame([record.dict() for record in records.records])
    data_preprocessed = preprocess_data(preprocessor, input_data)
    predictions = model.predict(data_preprocessed)
    return {"predictions": predictions.tolist()}

def preprocess_data(preprocessor, data: pd.DataFrame) -> pd.DataFrame:
    data_processed = preprocessor.transform(data)
    feature_names = preprocessor.get_feature_names_out()
    data_df = pd.DataFrame(data_processed, columns=feature_names)
    return data_df


