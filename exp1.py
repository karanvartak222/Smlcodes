import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')

print("\nDescriptive Statistics:")
print(df.describe())

numeric_df = df.select_dtypes(include=['number'])

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# First 6 numeric columns
columns = numeric_df.columns[:6]

plt.figure(figsize=(15, 10))

for i, col in enumerate(columns):
    plt.subplot(2, 3, i + 1)
    plt.hist(df[col], bins=20, color='skyblue', edgecolor='black')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()



Age,Study_Hours,Attendance,Assignments,Internal_Marks,Final_Marks
18,2,65,5,42,48
19,3,70,6,48,55
18,4,75,7,55,62
20,5,80,8,62,70
19,6,85,8,68,76
21,7,90,9,75,84
20,3,68,6,50,57
18,5,78,7,60,68
21,8,92,10,82,90
19,4,74,7,56,64
20,6,84,8,70,78
22,7,88,9,76,85
18,2,60,5,40,45
21,5,79,7,63,72
20,8,95,10,85,93
19,3,69,6,49,56
22,6,86,9,72,81
18,4,76,7,57,65
21,7,91,9,78,87
20,5,82,8,66,74
