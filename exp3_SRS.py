import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Create Student Performance Dataset

data = {
    "student_id": range(1, 21),

    "gender": [
        "Male", "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Male", "Female",
        "Male", "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Male", "Female"
    ],

    "study_hours": [
        4, 6, 2, 5, 7,
        3, 6, 4, 2, 7,
        5, 3, 6, 4, 2,
        7, 5, 3, 6, 4
    ],

    "attendance": [
        85, 92, 70, 88, 95,
        75, 90, 82, 68, 96,
        86, 73, 91, 84, 69,
        94, 87, 76, 89, 81
    ],

    "marks": [
        78, 88, 55, 82, 91,
        65, 85, 76, 52, 94,
        80, 62, 87, 74, 50,
        92, 79, 64, 86, 73
    ],

    "grade": [
        "B", "A", "C", "B", "A",
        "C", "A", "B", "C", "A",
        "B", "C", "A", "B", "C",
        "A", "B", "C", "A", "B"
    ],

    "city": [
        "Mumbai", "Vasai", "Thane", "Mumbai", "Vasai",
        "Thane", "Mumbai", "Vasai", "Thane", "Mumbai",
        "Vasai", "Thane", "Mumbai", "Vasai", "Thane",
        "Mumbai", "Vasai", "Thane", "Mumbai", "Vasai"
    ]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# ---------------------------------
# 1. Simple Random Sampling (20%)
# ---------------------------------

srs_sample = df.sample(frac=0.2, random_state=42)

print("\nSimple Random Sample:")
print(srs_sample)

# ---------------------------------
# 2. Stratified Sampling
# ---------------------------------

train, stratified_sample = train_test_split(
    df,
    test_size=0.2,
    stratify=df["grade"],
    random_state=42
)

print("\nStratified Sample:")
print(stratified_sample)

# ---------------------------------
# 3. Cluster Sampling
# ---------------------------------

clusters = df["city"].unique()

np.random.seed(42)
selected_cluster = np.random.choice(clusters)

cluster_sample = df[df["city"] == selected_cluster]

print("\nSelected Cluster:", selected_cluster)

print("\nCluster Sample:")
print(cluster_sample)

# ---------------------------------
# Result
# ---------------------------------

print("\nResult:")
print("Different samples were generated successfully.")
