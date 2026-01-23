import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit, Probit, Poisson
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("ECON 5140 - HOMEWORK 1")
print("Part A: Generalized Linear Models")
print("Part B: Time Series Decomposition")
print("=" * 70)

# ====================================================================
# DATASET 1: CUSTOMER PURCHASE DATA (for GLM analysis)
# ====================================================================
print(" \n" + "=" * 70)
print("DATASET 1: Customer Purchase Behavior")
print("=" * 70)

n_customers = 1000

# Generate customer features
age = np.random.normal(35, 10, n_customers)
income = np.random.normal(50, 15, n_customers)  # in thousands
time_on_site = np.random.gamma(2, 3, n_customers)  # in minutes

# True relationship (latent variable)
z = -3 + 0.05 * age + 0.04 * income + 0.15 * time_on_site + np.random.normal(0, 1, n_customers)

# Generate binary outcome (Purchase: 1=Yes, 0=No)
purchase = (z > 0).astype(int)

# Create DataFrame
df_customers = pd.DataFrame(
    {
        "Age": age,
        "Income": income,
        "TimeOnSite": time_on_site,
        "Purchase": purchase,
    }
)

print(f"Number of customers: {len(df_customers)}")
print(f"Purchase rate: {df_customers['Purchase'].mean():.2%}")
print(" \nFirst 5 rows:")
print(df_customers.head())

# ====================================================================
# DATASET 2: E-COMMERCE SALES TIME SERIES
# ====================================================================
print(" \n" + "=" * 70)
print("DATASET 2: E-commerce Daily Sales")
print("=" * 70)

# Create 2 years of daily data
dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
n_days = len(dates)
t = np.arange(n_days)

# Components
trend = 1000 + 2 * t + 0.01 * t**2
yearly_seasonal = 200 * np.sin(2 * np.pi * t / 365) + 150 * np.cos(2 * np.pi * t / 365)
weekly_seasonal = 100 * np.sin(2 * np.pi * t / 7)

# Special events
special_events = np.zeros(n_days)
for year in [2024, 2025]:
    # Black Friday
    bf_date = pd.Timestamp(f"{year}-11-24")
    bf_idx = dates == bf_date
    special_events[bf_idx] = 800

    # Christmas
    xmas_idx = (dates >= f"{year}-12-20") & (dates <= f"{year}-12-25")
    special_events[xmas_idx] = 400

# Random noise
noise = np.random.normal(0, 50, n_days)

# Combine components
sales = trend + yearly_seasonal + weekly_seasonal + special_events + noise
sales = np.maximum(sales, 0)

# Create DataFrame
df_sales = pd.DataFrame(
    {
        "Date": dates,
        "Sales": sales,
        "DayOfWeek": dates.dayofweek,
        "Month": dates.month,
        "IsWeekend": dates.dayofweek >= 5,
    }
)
df_sales.set_index("Date", inplace=True)

print(f"Date range: {df_sales.index[0].date()} to {df_sales.index[-1].date()}")
print(f"Number of days: {len(df_sales)}")
print(" \nSales Statistics:")
print(df_sales["Sales"].describe())

# ====================================================================
# PART A: GENERALIZED LINEAR MODELS
# ====================================================================
print(" \n" + "=" * 70)
print("PART A: GENERALIZED LINEAR MODELS")
print("=" * 70)

# --------------------------------------------------------------------
# A1: Exploratory Data Analysis (GLM)
# --------------------------------------------------------------------
print(" \n" + " -" * 70)
print("A1: Exploratory Data Analysis")
print(" -" * 70)

# 1. Create box plots comparing Age, Income, and TimeOnSite
#    between purchasers and non-purchasers
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["Age", "Income", "TimeOnSite"]):
    data = [
        df_customers.loc[df_customers["Purchase"] == 0, col],
        df_customers.loc[df_customers["Purchase"] == 1, col],
    ]
    ax.boxplot(data, labels=["No Purchase", "Purchase"])
    ax.set_title(f"{col} by Purchase")
    ax.set_ylabel(col)
plt.tight_layout()
plt.show()

# 2. Calculate and print mean values for each group
group_means = df_customers.groupby("Purchase")[["Age", "Income", "TimeOnSite"]].mean()
print("\nMean values by Purchase group:")
print(group_means)

# 3. Create a correlation matrix heatmap for the features
corr = df_customers[["Age", "Income", "TimeOnSite"]].corr()
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.columns)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
fig.colorbar(im, ax=ax)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# A2: Linear Probability Model (LPM)
# --------------------------------------------------------------------
print(" \n" + " -" * 70)
print("A2: Linear Probability Model")
print(" -" * 70)

# 1. Fit OLS model: Purchase ~ Age + Income + TimeOnSite
X = sm.add_constant(df_customers[["Age", "Income", "TimeOnSite"]])
y = df_customers["Purchase"]
lpm_model = sm.OLS(y, X).fit()

# 2. Print regression summary
print(lpm_model.summary())

# 3. Calculate predicted probabilities
lpm_pred = lpm_model.predict(X)
invalid_mask = (lpm_pred < 0) | (lpm_pred > 1)
invalid_pct = invalid_mask.mean() * 100
print(f"Invalid predictions outside [0, 1]: {invalid_mask.sum()} ({invalid_pct:.2f}%)")

# 4. Create histogram of predicted probabilities
plt.figure(figsize=(6, 4))
plt.hist(lpm_pred, bins=30, alpha=0.7, color="steelblue")
plt.axvline(0, color="red", linestyle="--")
plt.axvline(1, color="red", linestyle="--")
plt.title("LPM Predicted Probabilities")
plt.xlabel("Predicted probability")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# A3: Logistic Regression
# --------------------------------------------------------------------
print(" \n" + " -" * 70)
print("A3: Logistic Regression")
print(" -" * 70)

# 1. Fit logistic regression
logit_model = sm.Logit(y, X).fit(disp=False)

# 2. Print summary and extract coefficients, odds ratios, p-values
print(logit_model.summary())
coefficients = logit_model.params
odds_ratios = np.exp(coefficients)
p_values = logit_model.pvalues
print("\nLogit coefficients:")
print(coefficients)
print("\nOdds ratios:")
print(odds_ratios)
print("\nP-values:")
print(p_values)

# 3. Interpret each coefficient
print("\nInterpretation of coefficients (log-odds):")
print(f"Age: Each additional year changes log-odds by {coefficients['Age']:.4f}.")
print(f"Income: Each $1k increase changes log-odds by {coefficients['Income']:.4f}.")
print(f"TimeOnSite: Each extra minute changes log-odds by {coefficients['TimeOnSite']:.4f}.")

# 4. Calculate predicted probabilities and histogram
logit_pred = logit_model.predict(X)
print(f"Predicted probabilities within [0, 1]: {((logit_pred >= 0) & (logit_pred <= 1)).all()}")
plt.figure(figsize=(6, 4))
plt.hist(logit_pred, bins=30, alpha=0.7, color="seagreen")
plt.title("Logit Predicted Probabilities")
plt.xlabel("Predicted probability")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# A4: Prediction for New Customers
# --------------------------------------------------------------------
print(" \n" + " -" * 70)
print("A4: Predictions for New Customers")
print(" -" * 70)

# New customers
new_customers = pd.DataFrame(
    {
        "Age": [25, 35, 45, 55],
        "Income": [30, 50, 70, 90],
        "TimeOnSite": [2, 5, 8, 10],
    }
)

# 1. Predict purchase probability for each new customer
new_X = sm.add_constant(new_customers)
new_probs = logit_model.predict(new_X)

# 2. Create a formatted table
results_table = new_customers.copy()
results_table["PredictedProb"] = new_probs
results_table["PredictedPurchase"] = (results_table["PredictedProb"] > 0.5).astype(int)
print("\nNew customer predictions:")
print(results_table)

# 3. Most likely to purchase
most_likely_idx = results_table["PredictedProb"].idxmax()
most_likely = results_table.loc[most_likely_idx]
print(
    f"\nMost likely to purchase: Customer {most_likely_idx} "
    f"(Prob={most_likely['PredictedProb']:.3f}) due to higher income/time on site."
)

# ====================================================================
# PART B: TIME SERIES ANALYSIS
# ====================================================================
print(" \n" + "=" * 70)
print("PART B: TIME SERIES ANALYSIS")
print("=" * 70)

# --------------------------------------------------------------------
# B1: Time Series Visualization
# --------------------------------------------------------------------
print(" \n" + " -" * 70)
print("B1: Time Series Visualization")
print(" -" * 70)

# 1. Time series plot of daily sales
plt.figure(figsize=(12, 4))
plt.plot(df_sales.index, df_sales["Sales"], color="navy")
plt.title("Daily Sales (2024-2025)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# 2. Seasonal subseries plots
plt.figure(figsize=(10, 4))
df_sales.boxplot(column="Sales", by="DayOfWeek")
plt.title("Sales by Day of Week")
plt.suptitle("")
plt.xlabel("Day of Week (0=Mon)")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
df_sales.boxplot(column="Sales", by="Month")
plt.title("Sales by Month")
plt.suptitle("")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# 3. Mean sales by day of week and month
mean_by_day = df_sales.groupby("DayOfWeek")["Sales"].mean()
mean_by_month = df_sales.groupby("Month")["Sales"].mean()
print("\nMean sales by day of week:")
print(mean_by_day)
print("\nMean sales by month:")
print(mean_by_month)

# 4. Patterns observation
print("\nObserved patterns: weekly cycles and stronger seasonal months with event spikes.")

# --------------------------------------------------------------------
# B2: Stationarity Assessment
# --------------------------------------------------------------------
print(" \n" + " -" * 70)
print("B2: Stationarity Check")
print(" -" * 70)

rolling_mean = df_sales["Sales"].rolling(window=30).mean()
rolling_std = df_sales["Sales"].rolling(window=30).std()

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(df_sales.index, df_sales["Sales"], color="black")
axes[0].set_title("Original Series")
axes[1].plot(df_sales.index, rolling_mean, color="blue")
axes[1].set_title("30-Day Rolling Mean")
axes[2].plot(df_sales.index, rolling_std, color="orange")
axes[2].set_title("30-Day Rolling Std")
plt.tight_layout()
plt.show()

first_6m = df_sales.loc["2024-01-01":"2024-06-30", "Sales"]
last_6m = df_sales.loc["2025-07-01":"2025-12-31", "Sales"]
print("\nFirst 6 months mean/std:", first_6m.mean(), first_6m.std())
print("Last 6 months mean/std:", last_6m.mean(), last_6m.std())
print("Stationarity assessment: mean/variance change over time, so series is not stationary.")

# --------------------------------------------------------------------
# B3: Autocorrelation Analysis
# --------------------------------------------------------------------
print(" \n" + " -" * 70)
print("B3: Autocorrelation Function")
print(" -" * 70)

plot_acf(df_sales["Sales"], lags=60)
plt.title("ACF of Sales (up to 60 lags)")
plt.tight_layout()
plt.show()

def autocorr_at_lag(series, lag):
    return np.corrcoef(series[:-lag], series[lag:])[0, 1]

lag1 = autocorr_at_lag(sales, 1)
lag7 = autocorr_at_lag(sales, 7)
lag30 = autocorr_at_lag(sales, 30)
print(f"Autocorrelation lag 1: {lag1:.3f}")
print(f"Autocorrelation lag 7: {lag7:.3f}")
print(f"Autocorrelation lag 30: {lag30:.3f}")
print("Interpretation: weekly seasonality should show higher lag-7 correlation; persistence decays with lag.")

# --------------------------------------------------------------------
# B4: STL Decomposition
# --------------------------------------------------------------------
print(" \n" + " -" * 70)
print("B4: STL Decomposition")
print(" -" * 70)

stl = STL(df_sales["Sales"], seasonal=7, robust=True)
result = stl.fit()

result.plot()
plt.tight_layout()
plt.show()

print("\nSTL analysis:")
print("- Trend: upward with acceleration due to quadratic term.")
print("- Seasonal: weekly oscillation driven by weekly sine component.")
print("- Remainder: spikes on Black Friday and Christmas dates.")

# --------------------------------------------------------------------
# B5: Remainder Diagnostics
# --------------------------------------------------------------------
print(" \n" + " -" * 70)
print("B5: Remainder Analysis")
print(" -" * 70)

# 1. Extract remainder
remainder = result.resid

# 2. Diagnostic plots
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
axes[0].plot(df_sales.index, remainder, color="purple")
axes[0].set_title("Remainder Over Time")
axes[1].hist(remainder, bins=30, color="gray", alpha=0.8)
axes[1].set_title("Remainder Histogram")
plot_acf(remainder.dropna(), lags=60, ax=axes[2])
axes[2].set_title("ACF of Remainder")
plt.tight_layout()
plt.show()

# 3. Statistical tests
print(f"Remainder mean: {remainder.mean():.3f}")
print(f"Remainder std: {remainder.std():.3f}")
normal_test = stats.normaltest(remainder.dropna())
print(f"Normality test (D'Agostino): statistic={normal_test.statistic:.3f}, p-value={normal_test.pvalue:.3f}")

# 4. Identify outliers
threshold = 3 * remainder.std()
outliers = remainder[remainder.abs() > threshold]
print(f"\nOutliers (|remainder| > 3*std): {len(outliers)}")
print(outliers.sort_values(ascending=False).head(10))
print("\nNotable dates in outliers (Black Friday, Christmas) if present:")
for date in outliers.index:
    if date.strftime("%m-%d") in {"11-24", "12-20", "12-21", "12-22", "12-23", "12-24", "12-25"}:
        print(date.date(), "remainder:", outliers.loc[date])
