# ================================
# 1. Imports
# ================================

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
# ================================
# 2. Load Data
# ================================
data = pd.DataFrame(pd.read_csv(
    "C:/Users/soghr/TravelTide_Project/Mastery_project/merged_data.csv"))
# data_features = pd.DataFrame(pd.read_csv(
#    'C:/Users/soghr/TravelTide_Project/Mastery_project/user_features.csv'))
# data = data.merge(
# data_features[['user_id', 'Ave_monetary_value']], on='user_id', how='left')
# ================================
# 3. Define Variables
# ================================
# changing format of date columns
date_columns = ['session_start', 'session_end', 'birthdate', 'sign_up_date',
                'departure_time', 'return_time', 'check_in_time',
                'check_out_time']
for col in date_columns:
    data[col] = pd.to_datetime(data[col], format='mixed', errors='coerce')
data['session_duration'] = (
    (data['session_end'] -
     data['session_start']).dt.total_seconds()
    / 60
)

data['age'] = (pd.to_datetime(
    '2023-01-05') - pd.to_datetime(data['birthdate'])).dt.days // 365

data['flight_booked'] = np.where(data['flight_booked'] == True, 1, 0)
data['flight_discount'] = np.where(data['flight_discount'] == True, 1, 0)
data["hotel_discount"] = np.where(data["hotel_discount"] == True, 1, 0)
data['hotel_booked'] = np.where(data['hotel_booked'] == True, 1, 0)
data['booked_any'] = np.where(
    (data['flight_booked'] == 1) | (data['hotel_booked'] == 1), 1, 0)
data['discount_any'] = np.where(
    (data['flight_discount'] == 1) | (data['hotel_discount'] == 1), 1, 0)
data['married'] = np.where(data['married'] == 1.0, 1, 0)
data['has_children'] = np.where(
    data['has_children'] == 1.0, 1, 0)

# Target variable
TARGET = "hotel_booked"   # or hotel_booked

# Treatment variable
TREATMENT = "hotel_discount"   # or hotel_discount

# Features (adjust based on your dataset)
FEATURES = [
    "session_duration",
    "page_clicks",
    "base_fare_usd",
    "age",
    "married",
    "has_children"
]
df = data[FEATURES + [TARGET, TREATMENT]].copy()
# Drop missing values (simple approach)
df = df.dropna(subset=FEATURES + [TARGET, TREATMENT])
# print(df.describe)
# ================================
# 4. Split Data
# ================================

df_treat = df[df[TREATMENT] == 1]
df_control = df[df[TREATMENT] == 0]

print("Treatment size:", len(df_treat))
print("Control size:", len(df_control))
# ================================
# 5. Train Models
# ================================

# --- Treatment Model ---
X_treat = df_treat[FEATURES]
y_treat = df_treat[TARGET]

model_treat = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

model_treat.fit(X_treat, y_treat)


# --- Control Model ---
X_control = df_control[FEATURES]
y_control = df_control[TARGET]

model_control = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

model_control.fit(X_control, y_control)
# ================================
# 6. Predict Probabilities
# ================================

X_all = df[FEATURES]

# Probability of booking WITH discount
p_treat = model_treat.predict_proba(X_all)[:, 1]
# print(p_treat)
# Probability of booking WITHOUT discount
p_control = model_control.predict_proba(X_all)[:, 1]

df["p_treat"] = p_treat
df["p_control"] = p_control
# ================================
# 7. Uplift Calculation
# ================================

df["uplift"] = df["p_treat"] - df["p_control"]

df[["p_treat", "p_control", "uplift"]].head()
# ================================
# 8. Uplift Segmentation
# ================================


def uplift_segment(u):
    if u > 0.05:
        return "Persuadable"
    elif u > -0.02:
        return "Sure Thing"
    elif u <= -0.02:
        return "Do Not Disturb"
    else:
        return "Lost Cause"


df["uplift_segment"] = df["uplift"].apply(uplift_segment)

df["uplift_segment"].value_counts()
# ================================
# 9. Compare Conversion by Segment
# ================================

segment_performance = df.groupby("uplift_segment")[TARGET].mean()

print(segment_performance)
# ================================
# 10. Uplift Distribution
# ================================

plt.figure()
plt.hist(df["uplift"], bins=50)
plt.title("Uplift Distribution")
plt.xlabel("Uplift")
plt.ylabel("Frequency")
plt.show()
# ================================
# 11. Targeting Strategy
# ================================

# Sort by uplift
df_sorted = df.sort_values(by="uplift", ascending=False)

# Top 20% users
top_20 = df_sorted.head(int(0.2 * len(df)))

print("Average uplift (top 20%):", top_20["uplift"].mean())
print("Conversion rate (top 20%):", top_20[TARGET].mean())
# ================================
# Uplift Curve (Incremental Gain)
# ================================

# Sort by predicted uplift (descending)
df_sorted = df.sort_values(by="uplift", ascending=False).reset_index(drop=True)

# Create bins (percentiles)
n_bins = 20
df_sorted["bin"] = pd.qcut(df_sorted.index, q=n_bins, labels=False)

uplift_values = []
population = []

for i in range(n_bins):
    subset = df_sorted[df_sorted["bin"] <= i]

    treat = subset[subset[TREATMENT] == 1]
    control = subset[subset[TREATMENT] == 0]

    # Avoid division by zero
    if len(treat) > 0 and len(control) > 0:
        uplift = treat[TARGET].mean() - control[TARGET].mean()
    else:
        uplift = 0

    uplift_values.append(uplift)
    population.append((i + 1) / n_bins)

# Plot
plt.figure()
plt.plot(population, uplift_values, marker='o')

plt.title("Uplift Curve (Incremental Gain)")
plt.xlabel("Proportion of Targeted Users")
plt.ylabel("Incremental Uplift")
plt.axhline(0)

plt.show()
# ================================
# Qini Curve
# ================================

# Sort by uplift
df_qini = df.sort_values(by="uplift", ascending=False).reset_index(drop=True)

# Total number of samples
N = len(df_qini)

# Cumulative calculations
cum_treated = np.cumsum(df_qini[TREATMENT])
cum_control = np.cumsum(1 - df_qini[TREATMENT])

cum_y_treated = np.cumsum(df_qini[TARGET] * df_qini[TREATMENT])
cum_y_control = np.cumsum(df_qini[TARGET] * (1 - df_qini[TREATMENT]))

# Avoid division by zero
epsilon = 1e-6

# Compute incremental gain (Qini)
uplift_curve = (
    cum_y_treated / (cum_treated + epsilon)
    - cum_y_control / (cum_control + epsilon)
) * (np.arange(1, N+1))

# Normalize x-axis (population %)
x = np.arange(1, N+1) / N

# ----------------
# Random baseline
# ----------------
overall_treat_rate = df[df[TREATMENT] == 1][TARGET].mean()
overall_control_rate = df[df[TREATMENT] == 0][TARGET].mean()

random_uplift = (overall_treat_rate - overall_control_rate) * np.arange(1, N+1)

# ----------------
# Plot
# ----------------
plt.figure()

plt.plot(x, uplift_curve, label="Model")
plt.plot(x, random_uplift, linestyle="--", label="Random")

plt.title("Qini Curve")
plt.xlabel("Proportion of Targeted Users")
plt.ylabel("Incremental Gain")
plt.legend()

plt.show()
# ================================
# Qini Coefficient
# ================================

# qini_model = np.trapz(uplift_curve, x)
# qini_random = np.trapz(random_uplift, x)

# qini_coef = qini_model - qini_random

# print("Qini Coefficient:", qini_coef)
# ================================

# Train on full data
model_base = RandomForestClassifier(
    n_estimators=100, max_depth=6, random_state=42)

model_base.fit(df[FEATURES], df[TARGET])

df["p_book"] = model_base.predict_proba(df[FEATURES])[:, 1]
df["p_treat"]     # booking with discount
df["p_control"]   # booking without discount
df["uplift"]      # difference
# ================================
# Decision Engine
# ================================


def assign_reward(row):

    # High uplift → needs discount
    if row["uplift"] > 0.05 and row["p_book"] < 0.7:
        return "Give Discount"

    # High probability anyway → no discount
    elif row["p_book"] >= 0.7:
        return "No Discount (Sure Thing)"

    # Negative uplift → avoid
    elif row["uplift"] < 0:
        return "Avoid (Do Not Disturb)"

    # Default
    else:
        return "No Action"


df["decision"] = df.apply(assign_reward, axis=1)
print(df["decision"].value_counts())
# Compare conversion rates by decision
df.groupby("decision")[TARGET].mean()
# ================================
REVENUE_PER_BOOKING = 100   # adjust based on your data
DISCOUNT_COST = 20          # e.g. average discount


def expected_value(row):

    if row["decision"] == "Give Discount":
        return row["p_treat"] * REVENUE_PER_BOOKING - DISCOUNT_COST

    else:
        return row["p_control"] * REVENUE_PER_BOOKING


df["expected_value"] = df.apply(expected_value, axis=1)

df['user_id'] = data['user_id']
df = df.round(2)
df.to_csv("uplift_decision_for_graphics.csv", index=False)
