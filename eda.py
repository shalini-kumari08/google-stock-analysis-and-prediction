
# =============================================================================
# FILE: eda.py
# PROJECT: Stock Price Analysis and Prediction using Data Analytics in Python
# DESCRIPTION: Exploratory Data Analysis (EDA) on Google Stock Dataset
# AUTHOR: TYBCA Semester 6 Mini Project
# =============================================================================

# ---- Import Required Libraries ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---- Create output folder for saving graphs ----
os.makedirs("graphs", exist_ok=True)

# =============================================================================
# SECTION 1: LOAD DATASET
# =============================================================================
print("=" * 60)
print("   STOCK PRICE ANALYSIS - EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# Load the CSV dataset into a pandas DataFrame
df = pd.read_csv("dataset/Google_stock_data_550_rows.csv")

# Convert 'Date' column to datetime format for proper handling
df['Date'] = pd.to_datetime(df['Date'])

# Sort data by Date (chronological order)
df = df.sort_values('Date').reset_index(drop=True)

print("\n[INFO] Dataset loaded successfully!")

# =============================================================================
# SECTION 2: BASIC DATASET INFORMATION
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 2: BASIC DATASET INFORMATION")
print("-" * 60)

# Display first 5 rows
print("\n>> HEAD (First 5 Rows):")
print(df.head())

# Display last 5 rows
print("\n>> TAIL (Last 5 Rows):")
print(df.tail())

# Display shape of dataset
print(f"\n>> SHAPE of Dataset (Rows x Columns): {df.shape}")

# Display column names
print(f"\n>> COLUMNS: {list(df.columns)}")

# Display data types of each column
print("\n>> DATA TYPES:")
print(df.dtypes)

# =============================================================================
# SECTION 3: HANDLE MISSING VALUES
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 3: MISSING VALUES ANALYSIS")
print("-" * 60)

# Check total missing values per column
print("\n>> Missing Values per Column:")
print(df.isnull().sum())

# Check percentage of missing values
missing_percent = (df.isnull().sum() / len(df)) * 100
print("\n>> Missing Value Percentage (%):")
print(missing_percent)

# Fill missing values with column mean (if any)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

print("\n[INFO] Missing values handled (filled with column mean if any).")
print("[INFO] Missing values after handling:", df.isnull().sum().sum())

# =============================================================================
# SECTION 4: STATISTICAL SUMMARY
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 4: STATISTICAL SUMMARY")
print("-" * 60)

# Descriptive statistics for all numeric columns
print("\n>> Descriptive Statistics (describe):")
print(df.describe())

# Additional stats
print(f"\n>> Mean of 'Close' Price: {df['Close'].mean():.4f}")
print(f">> Median of 'Close' Price: {df['Close'].median():.4f}")
print(f">> Standard Deviation of 'Close': {df['Close'].std():.4f}")
print(f">> Min 'Close' Price: {df['Close'].min():.4f}")
print(f">> Max 'Close' Price: {df['Close'].max():.4f}")

# =============================================================================
# SECTION 5: CORRELATION MATRIX
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 5: CORRELATION MATRIX")
print("-" * 60)

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])

# Calculate correlation matrix
correlation = numeric_df.corr()
print("\n>> Correlation Matrix:")
print(correlation)

# =============================================================================
# SECTION 6: DATA VISUALIZATIONS
# =============================================================================
print("\n" + "-" * 60)
print("SECTION 6: DATA VISUALIZATIONS")
print("-" * 60)

# ---- Set global plot style ----
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 5)

# ---- PLOT 1: Line Chart - Closing Price Over Time ----
plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Close'], color='steelblue', linewidth=1.5, label='Close Price')
plt.title("Google Stock Closing Price Over Time", fontsize=14, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/01_closing_price_over_time.png", dpi=150)
plt.show()
print("[SAVED] Graph 1: Closing Price Over Time -> graphs/01_closing_price_over_time.png")

# ---- PLOT 2: Histogram - Distribution of Close Price ----
plt.figure(figsize=(10, 5))
plt.hist(df['Close'], bins=30, color='coral', edgecolor='black', alpha=0.85)#alpha is used for transparency
plt.title("Histogram - Distribution of Close Price", fontsize=14, fontweight='bold')
plt.xlabel("Close Price (USD)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("graphs/02_histogram_close_price.png", dpi=150)
plt.show()
print("[SAVED] Graph 2: Histogram -> graphs/02_histogram_close_price.png")

# ---- PLOT 3: Boxplot - Detecting Outliers in All Numeric Features ----
'''plt.figure(figsize=(12, 6))
numeric_df_box = df[['Open', 'High', 'Low', 'Close']]
numeric_df_box.boxplot(patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='navy'),
                        medianprops=dict(color='red', linewidth=2))
plt.title("Boxplot - Open, High, Low, Close Prices", fontsize=14, fontweight='bold')
plt.ylabel("Price (USD)")
plt.tight_layout()
plt.savefig("graphs/03_boxplot_prices.png", dpi=150)
plt.show()
print("[SAVED] Graph 3: Boxplot -> graphs/03_boxplot_prices.png")
'''
# ---- PLOT 4: Heatmap - Correlation Matrix ----

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, linecolor='gray', square=True)
plt.title("Heatmap - Correlation Matrix", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("graphs/04_heatmap_correlation.png", dpi=150)
plt.show()
print("[SAVED] Graph 4: Heatmap -> graphs/04_heatmap_correlation.png")


# ---- PLOT 5: Pairplot - Relationships Between Features ----
'''
print("\n[INFO] Generating Pairplot (this may take a few seconds)...")
pair_cols = ['Open', 'High', 'Low', 'Close']
pairplot_fig = sns.pairplot(df[pair_cols], diag_kind='kde',
                             plot_kws={'alpha': 0.5, 'color': 'steelblue'})
pairplot_fig.fig.suptitle("Pairplot - Stock Price Features", y=1.02,
                            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("graphs/05_pairplot_features.png", dpi=120)
plt.show()
print("[SAVED] Graph 5: Pairplot -> graphs/05_pairplot_features.png")
'''

# ---- PLOT 6: Volume Bar Chart ----
plt.figure(figsize=(12, 4))
plt.bar(df['Date'], df['Volume'], color='red', alpha=0.7, width=20)
plt.title("Google Stock Trading Volume Over Time", fontsize=14, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Volume")
plt.tight_layout()
plt.savefig("graphs/06_volume_over_time.png", dpi=150)
plt.show()
print("[SAVED] Graph 6: Volume Chart -> graphs/06_volume_over_time.png")

# =============================================================================
# SECTION 7: EDA CONCLUSION
# =============================================================================
print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)
print(">> All graphs have been saved in the 'graphs/' folder.")
print(">> Dataset Shape:", df.shape)
print(">> Date Range:", df['Date'].min().date(), "to", df['Date'].max().date())
print(">> Average Close Price: $", round(df['Close'].mean(), 2))
print(">> Max Close Price:     $", round(df['Close'].max(), 2))
print(">> Min Close Price:     $", round(df['Close'].min(), 2))
print("=" * 60)
