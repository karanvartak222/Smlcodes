import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ------------------------------------------
# DATASET
# ------------------------------------------

data = {
    'Study_Hours': [2, 4, 5, np.nan, 6, 7, 8, 20],
    'Gender': ['Male', 'Female', 'Male', 'Female',
               'Male', 'Female', 'Male', 'Female'],
    'Marks': [45, 55, 62, 68, np.nan, 78, 85, 150]
}

# Create DataFrame
df = pd.DataFrame(data)

# ------------------------------------------
# ORIGINAL DATASET
# ------------------------------------------

print("ORIGINAL DATASET")
print(df)

# ------------------------------------------
# CHECK MISSING VALUES
# ------------------------------------------

print("\nMISSING VALUES")
print(df.isnull().sum())

# ------------------------------------------
# HANDLE MISSING VALUES
# ------------------------------------------

df['Study_Hours'] = df['Study_Hours'].fillna(
    df['Study_Hours'].mean()
)

df['Marks'] = df['Marks'].fillna(
    df['Marks'].mean()
)

print("\nAFTER HANDLING MISSING VALUES")
print(df)

# ------------------------------------------
# ENCODING
# ------------------------------------------

df = pd.get_dummies(
    df,
    columns=['Gender'],
    dtype=int
)

print("\nAFTER ENCODING")
print(df)

# ------------------------------------------
# OUTLIER DETECTION USING IQR
# ------------------------------------------

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = ((df < lower) | (df > upper))

print("\nOUTLIERS")
print(outliers.sum())

# ------------------------------------------
# FEATURE SCALING
# ------------------------------------------

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(
    scaled_data,
    columns=df.columns
)

print("\nAFTER FEATURE SCALING")
print(scaled_df)

# ------------------------------------------
# FINAL PREPROCESSED DATASET
# ------------------------------------------

print("\nFINAL PREPROCESSED DATASET")
print(scaled_df)
