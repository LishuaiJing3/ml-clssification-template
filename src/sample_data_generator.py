#%%
import pandas as pd
import numpy as np


# Generate sample data
n_samples = 1000
data = {
    'Age': np.random.randint(18, 65, n_samples),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Location': np.random.choice(['City', 'Suburb', 'Rural'], n_samples),
    'Account_Type': np.random.choice(['Savings', 'Checking'], n_samples),
    'Tenure': np.random.randint(1, 11, n_samples),
    'Contract_Status': np.random.choice(['Month-to-Month', '1-Year', '2-Year'], n_samples),
    'Income': np.random.uniform(20000, 100000, n_samples),
    'Spending_Score': np.random.uniform(1, 10, n_samples),
    'Churn': np.random.choice([0, 1], n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('data/sample_data.csv', index=False)

# %%
