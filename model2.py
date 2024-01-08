import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

df = pd.read_csv("tidy_data.csv")

# Removing the intercept
# No need to include interactions
X = df.drop(columns=["Score"])
y = np.array(df["Score"])

# One-hot encode non-numerical columns
X = pd.get_dummies(X)

# Get column names before converting to numpy array
column_names = X.columns

# Convert to numpy array
X = np.array(X)

# 10-fold
kf = KFold(n_splits=10, shuffle=True, random_state=42)

lambdas = [2, 4, 6, 8, 10, 12, 15, 18, 20, 24, 50, 100]
fitted = np.zeros((len(y), len(lambdas)))

for train_idx, test_idx in kf.split(X):
    for i, lam in enumerate(lambdas):
        model = Lasso(alpha=lam, max_iter=10000)
        model.fit(X[train_idx, :], y[train_idx])
        fitted[test_idx, i] = model.predict(X[test_idx, :])

# Calculate mean squared error
mse = np.mean((y[:, np.newaxis] - fitted) ** 2, axis=0)

result = np.vstack((lambdas, mse)).T
print(result)

# choose lambda=15
selected_lambda = 15

for train_idx, test_idx in kf.split(X):
    model = Lasso(alpha=selected_lambda, max_iter=10000)
    model.fit(X[train_idx, :], y[train_idx])

    # Get non-zero coefficients and corresponding variable names
    non_zero_coeffs = model.coef_[model.coef_ != 0]
    non_zero_variables = column_names[model.coef_ != 0]

    # Print non-zero coefficients and corresponding variable names
    for variable, coefficient in zip(non_zero_variables, non_zero_coeffs):
        print(
            f"Variable: {variable}, Coefficient for lambda={selected_lambda}: {coefficient}"
        )
