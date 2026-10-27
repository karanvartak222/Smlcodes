# ============================================================
# EXPERIMENT 08
# Linear Regression with Model Diagnostics
# ============================================================

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ============================================================
# 2. Create Changed Dataset
# ============================================================

data = {
    'Study_Hours': [
        2, 3, 4, 5, 6,
        7, 8, 9, 3, 5,
        6, 7, 8, 10, 12
    ],

    'Attendance': [
        60, 65, 70, 72, 75,
        78, 80, 82, 68, 74,
        76, 79, 83, 88, 92
    ],

    'Assignment_Score': [
        45, 50, 55, 60, 62,
        68, 70, 75, 52, 61,
        65, 69, 74, 82, 95
    ],

    'Exam_Marks': [
        42, 48, 53, 58, 64,
        69, 74, 78, 50, 59,
        65, 70, 76, 85, 98
    ]
}

df = pd.DataFrame(data)

print("=" * 60)
print("DATASET")
print("=" * 60)
print(df)


# ============================================================
# 3. Define Independent and Dependent Variables
# ============================================================

X = df[['Study_Hours', 'Attendance', 'Assignment_Score']]
y = df['Exam_Marks']

print("\n" + "=" * 60)
print("INDEPENDENT VARIABLES (X)")
print("=" * 60)
print(X)

print("\nDependent Variable (Y): Exam_Marks")


# ============================================================
# 4. Fit Linear Regression Model
# ============================================================

model = LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)


# ============================================================
# 5. Model Evaluation
# ============================================================

r2 = r2_score(y, y_pred)

mse = mean_squared_error(y, y_pred)

rmse = np.sqrt(mse)

print("\n" + "=" * 60)
print("LINEAR REGRESSION MODEL")
print("=" * 60)

print("Intercept:", model.intercept_)

print("Coefficients:", model.coef_)

print("R2 Score:", r2)

print("MSE:", mse)

print("RMSE:", rmse)


# ============================================================
# 6. Regression Equation
# ============================================================

print("\nRegression Equation:")

print(
    "Exam_Marks =",
    round(model.intercept_, 4),
    "+",
    round(model.coef_[0], 4), "(Study_Hours)",
    "+",
    round(model.coef_[1], 4), "(Attendance)",
    "+",
    round(model.coef_[2], 4), "(Assignment_Score)"
)


# ============================================================
# 7. Calculate VIF
# ============================================================

X_vif = sm.add_constant(X)

vif = pd.DataFrame()

vif["Variable"] = X_vif.columns

vif["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]

print("\n" + "=" * 60)
print("VIF RESULTS")
print("=" * 60)

print(vif)


# ============================================================
# 8. Calculate Residuals
# ============================================================

residuals = y - y_pred

diagnostics = pd.DataFrame({
    'Actual': y,
    'Predicted': y_pred,
    'Residual': residuals
})

print("\n" + "=" * 60)
print("RESIDUALS")
print("=" * 60)

print(diagnostics)


# ============================================================
# 9. Residual vs Fitted Values Plot
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=y_pred,
    y=residuals
)

plt.axhline(
    y=0,
    color='red',
    linestyle='--'
)

plt.xlabel("Fitted / Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")

plt.show()


# ============================================================
# 10. Statsmodels Regression Model
# ============================================================

X_sm = sm.add_constant(X)

ols_model = sm.OLS(y, X_sm).fit()

print("\n" + "=" * 60)
print("OLS REGRESSION SUMMARY")
print("=" * 60)

print(ols_model.summary())


# ============================================================
# 11. Calculate Cook's Distance
# ============================================================

influence = ols_model.get_influence()

cooks_distance = influence.cooks_distance[0]

diagnostics['Cook_Distance'] = cooks_distance


print("\n" + "=" * 60)
print("COOK'S DISTANCE")
print("=" * 60)

for i, value in enumerate(cooks_distance):
    print(
        "Observation:",
        i + 1,
        "Cook's Distance:",
        round(value, 4)
    )


# ============================================================
# 12. Plot Cook's Distance
# ============================================================

plt.figure(figsize=(10, 5))

plt.stem(
    range(len(cooks_distance)),
    cooks_distance
)

plt.xlabel("Observation Number")
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance")

plt.show()


# ============================================================
# 13. Identify Influential Observations
# ============================================================

n = len(df)

threshold = 4 / n

diagnostics['Influential'] = (
    diagnostics['Cook_Distance'] > threshold
)

print("\n" + "=" * 60)
print("COOK'S DISTANCE THRESHOLD")
print("=" * 60)

print("Number of observations:", n)

print("Threshold (4/n):", threshold)


print("\n" + "=" * 60)
print("INFLUENTIAL OBSERVATIONS")
print("=" * 60)

found = False

for i in range(n):

    if cooks_distance[i] > threshold:

        found = True

        print(
            "Observation:",
            i + 1,
            "| Cook's Distance:",
            round(cooks_distance[i], 4)
        )

if not found:
    print("No influential observations found above the threshold.")


# ============================================================
# 14. Final Diagnostic Results
# ============================================================

print("\n" + "=" * 60)
print("FINAL DIAGNOSTIC RESULTS")
print("=" * 60)

print(diagnostics)


# ============================================================
# 15. Simple Interpretation
# ============================================================

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print("""
1. R2 Score:
   R2 indicates how much variation in Exam_Marks
   is explained by Study_Hours, Attendance and
   Assignment_Score.

2. VIF:
   VIF is used to detect multicollinearity among
   the independent variables.

3. Residual Plot:
   Residuals should be randomly distributed around
   zero without a clear pattern.

4. Cook's Distance:
   Cook's distance identifies observations that have
   a strong influence on the regression model.

5. Influential Observations:
   Observations having Cook's Distance greater than
   4/n are flagged for further investigation.
""")

print("\nExperiment completed successfully.")
