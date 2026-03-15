
# =============================================================================
# FILE: simple_linear_regression.py
# PROJECT: Stock Price Analysis and Prediction using Data Analytics in Python
# DESCRIPTION: Simple Linear Regression - Predict 'Close' using 'Open' price
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
print("   SIMPLE LINEAR REGRESSION")
print("   Independent Variable: Open  |  Dependent Variable: Close")
print("=" * 60)

# Load dataset
df = pd.read_csv("dataset/Google_stock_data_550_rows.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Drop rows with missing values (if any)
df.dropna(inplace=True)

print(f"\n[INFO] Dataset loaded. Total records: {len(df)}")

# =============================================================================
# SECTION 2: FEATURE SELECTION
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 2: FEATURE SELECTION")
print("-" * 60)

# Independent variable (X): Open Price
# Dependent variable (y): Close Price
X = df[['Open']]   # Feature - must be 2D for sklearn
y = df['Close']    # Target variable

print(f"\n>> Independent Variable (X): Open Price")
print(f">> Dependent Variable   (y): Close Price")
print(f">> Total Samples: {len(X)}")

# =============================================================================
# SECTION 3: SPLIT DATASET INTO TRAIN AND TEST
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 3: TRAIN-TEST SPLIT (80% Train | 20% Test)")
print("-" * 60)

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n>> Training samples : {len(X_train)}")
print(f">> Testing  samples : {len(X_test)}")

# =============================================================================
# SECTION 4: TRAIN THE SIMPLE LINEAR REGRESSION MODEL
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 4: TRAINING THE MODEL")
print("-" * 60)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Display model parameters
print(f"\n>> Model Intercept (b0)  : {model.intercept_:.6f}")
print(f">> Model Coefficient (b1): {model.coef_[0]:.6f}")
print(f"\n   Regression Equation:")
print(f"   Close = {model.intercept_:.4f} + ({model.coef_[0]:.4f} × Open)")

# =============================================================================
# SECTION 5: PREDICTIONS
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 5: PREDICTIONS")
print("-" * 60)

# Predict on test data
y_pred = model.predict(X_test)

# Show a comparison of actual vs predicted values
comparison = pd.DataFrame({
    'Actual Close': y_test.values[:10],
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

# Calculate evaluation metrics
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\n>> R² Score (Accuracy)          : {r2:.6f}  ({r2*100:.2f}%)")
print(f">> Mean Absolute Error  (MAE)   : {mae:.6f}")
print(f">> Mean Squared Error   (MSE)   : {mse:.6f}")
print(f">> Root Mean Sq. Error  (RMSE)  : {rmse:.6f}")

print("\n[INTERPRETATION]")
print(f"   R² = {r2:.4f} means the model explains {r2*100:.2f}% of the")
print("   variance in Close Price using Open Price.")

# =============================================================================
# SECTION 7: VISUALIZATION - REGRESSION LINE
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 7: PLOTTING REGRESSION LINE")
print("-" * 60)

sns.set_theme(style="darkgrid")

# ---- PLOT 1: Scatter Plot + Regression Line ----
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='steelblue', alpha=0.6, label='Actual Values', s=30)
plt.plot(X_test, y_pred, color='red', linewidth=2.5, label='Regression Line')
plt.title("Simple Linear Regression\nOpen Price vs Close Price", fontsize=14, fontweight='bold')
plt.xlabel("Open Price (USD)")
plt.ylabel("Close Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/07_simple_linear_regression.png", dpi=150)
plt.show()
print("[SAVED] Graph 7: Regression Line -> graphs/07_simple_linear_regression.png")

# ---- PLOT 2: Residual Plot ----
residuals = y_test.values - y_pred
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, color='mediumpurple', alpha=0.6, s=30)
plt.axhline(y=0, color='red', linewidth=2, linestyle='--')
plt.title("Residual Plot - Simple Linear Regression", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.tight_layout()
plt.savefig("graphs/08_slr_residual_plot.png", dpi=150)
plt.show()
print("[SAVED] Graph 8: Residual Plot -> graphs/08_slr_residual_plot.png")

# ---- PLOT 3: Actual vs Predicted ----
plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test)), y_test.values, color='steelblue', label='Actual', alpha=0.8)
plt.plot(range(len(y_pred)), y_pred,         color='red',       label='Predicted', alpha=0.8, linestyle='--')
plt.title("Actual vs Predicted - Close Price (Test Set)", fontsize=14, fontweight='bold')
plt.xlabel("Sample Index")
plt.ylabel("Close Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/09_slr_actual_vs_predicted.png", dpi=150)
plt.show()
print("[SAVED] Graph 9: Actual vs Predicted -> graphs/09_slr_actual_vs_predicted.png")

# =============================================================================
# SECTION 8: SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("SIMPLE LINEAR REGRESSION COMPLETE")
print("=" * 60)
print(f"  Equation : Close = {model.intercept_:.4f} + {model.coef_[0]:.4f} × Open")
print(f"  R² Score : {r2:.4f} ({r2*100:.2f}%)")
print(f"  MAE      : {mae:.4f}")
print(f"  MSE      : {mse:.4f}")
print(f"  RMSE     : {rmse:.4f}")
print("=" * 60)
