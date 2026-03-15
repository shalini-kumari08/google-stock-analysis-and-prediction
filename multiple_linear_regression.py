
# =============================================================================
# FILE: multiple_linear_regression.py
# PROJECT: Stock Price Analysis and Prediction using Data Analytics in Python
# DESCRIPTION: Multiple Linear Regression - Predict 'Close' using Open, High, Low, Volume
# AUTHOR: TYBCA Semester 6 Mini Project
# =============================================================================

# ---- Import Required Libraries ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---- Create output folder for saving graphs ----
os.makedirs("graphs", exist_ok=True)

# =============================================================================
# SECTION 1: LOAD AND PREPARE DATASET
# =============================================================================
print("=" * 60)
print("   MULTIPLE LINEAR REGRESSION")
print("   Predicting 'Close' Price from Open, High, Low, Volume")
print("=" * 60)

# Load dataset
df = pd.read_csv("dataset/Google_stock_data_550_rows.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Drop rows with missing values
df.dropna(inplace=True)

print(f"\n[INFO] Dataset loaded. Total records: {len(df)}")

# =============================================================================
# SECTION 2: FEATURE SELECTION (Multiple Features)
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 2: FEATURE SELECTION")
print("-" * 60)

# Define independent variables (multiple features)
feature_columns = ['Open', 'High', 'Low', 'Volume']

# Independent variables (X)
X = df[feature_columns]

# Dependent variable (y): Close Price
y = df['Close']

print(f"\n>> Independent Variables (X): {feature_columns}")
print(f">> Dependent Variable   (y): Close")
print(f">> Total Samples: {len(X)}")
print(f"\n>> Feature Data Preview:")
print(X.head())

# =============================================================================
# SECTION 3: TRAIN-TEST SPLIT
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 3: TRAIN-TEST SPLIT (80% Train | 20% Test)")
print("-" * 60)

# Split dataset: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n>> Training samples : {len(X_train)}")
print(f">> Testing  samples : {len(X_test)}")

# =============================================================================
# SECTION 4: TRAIN MULTIPLE LINEAR REGRESSION MODEL
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 4: TRAINING THE MODEL")
print("-" * 60)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Display intercept
print(f"\n>> Model Intercept (b0): {model.intercept_:.6f}")

# Display coefficients for each feature
print("\n>> Model Coefficients:")
coeff_df = pd.DataFrame({
    'Feature'     : feature_columns,
    'Coefficient' : model.coef_
})
print(coeff_df.to_string(index=False))

# Display the full regression equation
print("\n>> Regression Equation:")
eq = f"Close = {model.intercept_:.4f}"
for feat, coef in zip(feature_columns, model.coef_):
    sign = "+" if coef >= 0 else "-"
    eq += f" {sign} ({abs(coef):.6f} × {feat})"
print(f"   {eq}")

# =============================================================================
# SECTION 5: PREDICTIONS
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 5: PREDICTIONS")
print("-" * 60)

# Predict on test set
y_pred = model.predict(X_test)

# Sample comparison
comparison = pd.DataFrame({
    'Actual Close'   : y_test.values[:10].round(4),
    'Predicted Close': y_pred[:10].round(4)
})
print("\n>> Sample Actual vs Predicted Values (first 10):")
print(comparison.to_string(index=False))

# =============================================================================
# SECTION 6: MODEL EVALUATION
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 6: MODEL EVALUATION METRICS")
print("-" * 60)

# Compute evaluation metrics
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\n>> R² Score (Accuracy)          : {r2:.6f}  ({r2*100:.2f}%)")
print(f">> Mean Absolute Error  (MAE)   : {mae:.6f}")
print(f">> Mean Squared Error   (MSE)   : {mse:.6f}")
print(f">> Root Mean Sq. Error  (RMSE)  : {rmse:.6f}")

print("\n[INTERPRETATION]")
print(f"   R² = {r2:.4f} means the model explains {r2*100:.2f}% of")
print("   variance in Close Price using all 4 features combined.")

# =============================================================================
# SECTION 7: FEATURE IMPORTANCE INTERPRETATION
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 7: FEATURE IMPORTANCE INTERPRETATION")
print("-" * 60)

# Use absolute value of coefficients as importance (normalized)
abs_coefs = np.abs(model.coef_)
importance = abs_coefs / abs_coefs.sum() * 100

importance_df = pd.DataFrame({
    'Feature'           : feature_columns,
    'Coefficient'       : model.coef_.round(6),
    'Abs Coefficient'   : abs_coefs.round(6),
    'Importance (%)'    : importance.round(2)
}).sort_values('Importance (%)', ascending=False)

print("\n>> Feature Importance Table (by Absolute Coefficient):")
print(importance_df.to_string(index=False))

# =============================================================================
# SECTION 8: VISUALIZATIONS
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 8: VISUALIZATIONS")
print("-" * 60)

sns.set_theme(style="darkgrid")

# ---- PLOT 1: Feature Importance Bar Chart ----
plt.figure(figsize=(9, 5))
colors = ['steelblue', 'coral', 'mediumpurple', 'mediumseagreen']
bars = plt.bar(importance_df['Feature'], importance_df['Importance (%)'],
               color=colors, edgecolor='black', alpha=0.85)
# Add value labels on top of bars
for bar, val in zip(bars, importance_df['Importance (%)']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{val:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.title("Feature Importance - Multiple Linear Regression\n(by Absolute Coefficient)", fontsize=13, fontweight='bold')
plt.xlabel("Feature")
plt.ylabel("Importance (%)")
plt.tight_layout()
plt.savefig("graphs/10_mlr_feature_importance.png", dpi=150)
plt.show()
print("[SAVED] Graph 10: Feature Importance -> graphs/10_mlr_feature_importance.png")

# ---- PLOT 2: Actual vs Predicted ----
plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test)), y_test.values, color='steelblue', label='Actual',    alpha=0.85)
plt.plot(range(len(y_pred)), y_pred,         color='red',       label='Predicted', alpha=0.85, linestyle='--')
plt.title("Actual vs Predicted - Close Price (MLR)", fontsize=14, fontweight='bold')
plt.xlabel("Sample Index")
plt.ylabel("Close Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/11_mlr_actual_vs_predicted.png", dpi=150)
plt.show()
print("[SAVED] Graph 11: Actual vs Predicted -> graphs/11_mlr_actual_vs_predicted.png")

# ---- PLOT 3: Perfect Fit Scatter (Actual vs Predicted) ----
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, color='steelblue', alpha=0.5, s=25, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Fit Line')
plt.title("Actual vs Predicted Scatter - MLR", fontsize=14, fontweight='bold')
plt.xlabel("Actual Close Price (USD)")
plt.ylabel("Predicted Close Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/12_mlr_scatter_actual_vs_predicted.png", dpi=150)
plt.show()
print("[SAVED] Graph 12: Scatter Plot -> graphs/12_mlr_scatter_actual_vs_predicted.png")

# ---- PLOT 4: Residual Plot ----
residuals = y_test.values - y_pred
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, color='coral', alpha=0.6, s=30)
plt.axhline(0, color='black', linewidth=2, linestyle='--')
plt.title("Residual Plot - Multiple Linear Regression", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Close Price")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig("graphs/13_mlr_residual_plot.png", dpi=150)
plt.show()
print("[SAVED] Graph 13: Residual Plot -> graphs/13_mlr_residual_plot.png")

# =============================================================================
# SECTION 9: SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("MULTIPLE LINEAR REGRESSION COMPLETE")
print("=" * 60)
print(f"  Features Used : {feature_columns}")
print(f"  R² Score      : {r2:.4f} ({r2*100:.2f}%)")
print(f"  MAE           : {mae:.4f}")
print(f"  MSE           : {mse:.4f}")
print(f"  RMSE          : {rmse:.4f}")
print("=" * 60)
