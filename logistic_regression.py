
# =============================================================================
# FILE: logistic_regression.py
# PROJECT: Stock Price Analysis and Prediction using Data Analytics in Python
# DESCRIPTION: Logistic Regression - Predict whether stock price will increase
# Binary Target: 1 = Price Increased (Close > Open), 0 = Price Decreased
# AUTHOR: TYBCA Semester 6 Mini Project
# =============================================================================

# ---- Import Required Libraries ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, roc_auc_score)
from sklearn.preprocessing import StandardScaler

# ---- Create output folder for saving graphs ----
os.makedirs("graphs", exist_ok=True)

# =============================================================================
# SECTION 1: LOAD DATASET
# =============================================================================
print("=" * 60)
print("   LOGISTIC REGRESSION")
print("   Target: Price Increased (1) or Decreased (0)")
print("=" * 60)

# Load dataset
df = pd.read_csv("dataset/Google_stock_data_550_rows.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Drop rows with missing values
df.dropna(inplace=True)

print(f"\n[INFO] Dataset loaded. Total records: {len(df)}")

# =============================================================================
# SECTION 2: CREATE BINARY TARGET COLUMN
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 2: CREATING BINARY TARGET VARIABLE")
print("-" * 60)

# Create binary target:
# 1 -> Price Increased  (Close > Open, i.e., market went up on that day)
# 0 -> Price Decreased  (Close <= Open, i.e., market went down or flat)
df['Price_Increased'] = (df['Close'] > df['Open']).astype(int)

print("\n>> Binary Target Column: 'Price_Increased'")
print("   1 = Close Price > Open Price  (Price INCREASED)")
print("   0 = Close Price <= Open Price (Price DECREASED or flat)")

# Show distribution of target
print("\n>> Target Variable Distribution:")
dist = df['Price_Increased'].value_counts()
print(f"   Class 1 (Price Increased) : {dist.get(1, 0)} records")
print(f"   Class 0 (Price Decreased) : {dist.get(0, 0)} records")
print(f"   Total                     : {len(df)} records")

# =============================================================================
# SECTION 3: FEATURE SELECTION
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 3: FEATURE SELECTION")
print("-" * 60)

# Features: Open, High, Low, Volume
feature_columns = ['Open', 'High', 'Low', 'Volume']
X = df[feature_columns]
y = df['Price_Increased']

print(f"\n>> Independent Variables (X): {feature_columns}")
print(f">> Dependent Variable   (y): Price_Increased (0 or 1)")

# =============================================================================
# SECTION 4: TRAIN-TEST SPLIT
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 4: TRAIN-TEST SPLIT (80% Train | 20% Test)")
print("-" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n>> Training samples : {len(X_train)}")
print(f">> Testing  samples : {len(X_test)}")

# =============================================================================
# SECTION 5: FEATURE SCALING (Standardization)
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 5: FEATURE SCALING (StandardScaler)")
print("-" * 60)

# Scale features for logistic regression (important for convergence)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n[INFO] Features scaled using StandardScaler.")
print("       This ensures all features are on the same scale.")

# =============================================================================
# SECTION 6: TRAIN LOGISTIC REGRESSION MODEL
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 6: TRAINING THE LOGISTIC REGRESSION MODEL")
print("-" * 60)

# Create and train Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print("\n[INFO] Model trained successfully!")
print(f">> Intercept: {model.intercept_[0]:.6f}")

# Show coefficients
coeff_df = pd.DataFrame({
    'Feature'     : feature_columns,
    'Coefficient' : model.coef_[0].round(6)
})
print("\n>> Model Coefficients:")
print(coeff_df.to_string(index=False))

# =============================================================================
# SECTION 7: PREDICTIONS
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 7: PREDICTIONS")
print("-" * 60)

# Predict class labels
y_pred = model.predict(X_test_scaled)

# Predict probabilities (needed for ROC curve)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Sample comparison
comparison = pd.DataFrame({
    'Actual'            : y_test.values[:10],
    'Predicted'         : y_pred[:10],
    'Probability (1)'   : y_prob[:10].round(4)
})
print("\n>> Sample Predictions (first 10):")
print(comparison.to_string(index=False))

# =============================================================================
# SECTION 8: MODEL EVALUATION
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 8: MODEL EVALUATION")
print("-" * 60)

# ---- Accuracy Score ----
accuracy = accuracy_score(y_test, y_pred)
print(f"\n>> Accuracy Score: {accuracy:.6f}  ({accuracy*100:.2f}%)")

# ---- Confusion Matrix ----
cm = confusion_matrix(y_test, y_pred)
print("\n>> Confusion Matrix:")
print(cm)
print("\n   Layout:")
print("   +-----------------------------+")
print("   |          | Predicted 0 | Predicted 1 |")
print("   +----------+-------------+-------------+")
print(f"   | Actual 0 |   TN = {cm[0][0]:3d}  |   FP = {cm[0][1]:3d}  |")
print(f"   | Actual 1 |   FN = {cm[1][0]:3d}  |   TP = {cm[1][1]:3d}  |")
print("   +-----------------------------+")

# ---- Classification Report ----
print("\n>> Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=['Decreased (0)', 'Increased (1)']))

# ---- ROC AUC Score ----
auc = roc_auc_score(y_test, y_prob)
print(f">> ROC AUC Score: {auc:.6f}")

# =============================================================================
# SECTION 9: VISUALIZATIONS
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 9: VISUALIZATIONS")
print("-" * 60)

sns.set_theme(style="darkgrid")

# ---- PLOT 1: Confusion Matrix Heatmap ----
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0\n(Decreased)', 'Predicted 1\n(Increased)'],
            yticklabels=['Actual 0\n(Decreased)', 'Actual 1\n(Increased)'],
            linewidths=0.5, linecolor='gray', annot_kws={"size": 14, "weight": "bold"})
plt.title("Confusion Matrix - Logistic Regression", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("graphs/14_logistic_confusion_matrix.png", dpi=150)
plt.show()
print("[SAVED] Graph 14: Confusion Matrix -> graphs/14_logistic_confusion_matrix.png")

# ---- PLOT 2: ROC Curve ----
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='steelblue', linewidth=2.5, label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1.5, label='Random Classifier')
plt.fill_between(fpr, tpr, alpha=0.10, color='steelblue')
plt.title("ROC Curve - Logistic Regression", fontsize=14, fontweight='bold')
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR / Recall)")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("graphs/15_logistic_roc_curve.png", dpi=150)
plt.show()
print("[SAVED] Graph 15: ROC Curve -> graphs/15_logistic_roc_curve.png")

# ---- PLOT 3: Target Class Distribution ----
plt.figure(figsize=(7, 5))
labels = ['Decreased (0)', 'Increased (1)']
counts = [dist.get(0, 0), dist.get(1, 0)]
colors = ['coral', 'mediumseagreen']
bars = plt.bar(labels, counts, color=colors, edgecolor='black', alpha=0.85)
for bar, val in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
             str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.title("Target Class Distribution\n(Price Increased vs Decreased)", fontsize=13, fontweight='bold')
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("graphs/16_logistic_class_distribution.png", dpi=150)
plt.show()
print("[SAVED] Graph 16: Class Distribution -> graphs/16_logistic_class_distribution.png")

# ---- PLOT 4: Prediction Probability Distribution ----
plt.figure(figsize=(9, 5))
plt.hist(y_prob[y_test == 0], bins=25, alpha=0.65, color='coral',
         label='Actual: Decreased (0)', edgecolor='black')
plt.hist(y_prob[y_test == 1], bins=25, alpha=0.65, color='mediumseagreen',
         label='Actual: Increased (1)', edgecolor='black')
plt.axvline(0.5, color='navy', linestyle='--', linewidth=2, label='Threshold = 0.5')
plt.title("Predicted Probability Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Probability of Increase (Class 1)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/17_logistic_probability_distribution.png", dpi=150)
plt.show()
print("[SAVED] Graph 17: Probability Distribution -> graphs/17_logistic_probability_distribution.png")

# =============================================================================
# SECTION 10: SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION COMPLETE")
print("=" * 60)
print(f"  Target Variable : Price_Increased (Binary: 0 or 1)")
print(f"  Features Used   : {feature_columns}")
print(f"  Accuracy Score  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  ROC AUC Score   : {auc:.4f}")
print(f"  Confusion Matrix:")
print(f"     TN={cm[0][0]}  FP={cm[0][1]}")
print(f"     FN={cm[1][0]}  TP={cm[1][1]}")
print("=" * 60)
