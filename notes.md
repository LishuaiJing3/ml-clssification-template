## TODO:

add dimension reduction for catogorical if diemnsion gets high or reduce dimensions. Need to see how to make shp work if we reduce domension when interpeting results. 

##
poetry add pandas-stubs to fix mypy issue

## mlflow
Start mlflow ui with: cd to project root, then run mlflow ui in the terminal

## fastapi

for some reason, uvicorn does not work on the root. Need to cd to where the inference code is and start server there. 

cd src/inference

uvicorn inference_fastapi_template:app --reload 
example to test the endpoint

{
  "Age": 31,
  "Gender": "Male",
  "Location": "City",
  "Account_Type": "Checking",
  "Tenure": 7,
  "Contract_Status": "2-Year",
  "Income": 67101.88800508474,
  "Spending_Score": 5.623615990693347
}

batch: 

{
  "records": [
    {
      "Age": 31,
      "Gender": "Male",
      "Location": "City",
      "Account_Type": "Checking",
      "Tenure": 7,
      "Contract_Status": "2-Year",
      "Income": 67101.88800508474,
      "Spending_Score": 5.623615990693347
    },
    {
      "Age": 29,
      "Gender": "Female",
      "Location": "Suburban",
      "Account_Type": "Savings",
      "Tenure": 3,
      "Contract_Status": "Month-to-Month",
      "Income": 85000,
      "Spending_Score": 7.2
    }
  ]
}
