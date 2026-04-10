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
    "C:/Users/soghr/TravelTide_Project/datasets/merged_data.csv"))
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
data['booked_both'] = np.where(
    (data['flight_booked'] == 1) & (data['hotel_booked'] == 1), 1, 0)
data['discount_any'] = np.where(
    (data['flight_discount'] == 1) | (data['hotel_discount'] == 1), 1, 0)

data['married'] = np.where(data['married'] == 1.0, 1, 0)
data['has_children'] = np.where(
    data['has_children'] == 1.0, 1, 0)

# Target variable
TARGET = "hotel_booked"

# Treatment variable
TREATMENT = "hotel_discount"

# Features
FEATURES = [
    "session_duration",
    'page_clicks',
    "age",
    "married",
    "has_children"
]
# Bring user_id in early (as suggested earlier)
df = data[FEATURES + [TARGET, TREATMENT, 'user_id']].copy()
# df['hotel_per_room_usd'] = df['hotel_per_room_usd'].fillna(
#    df['hotel_per_room_usd'].median())

# Drop missing values (simple approach)
# df = df.dropna(subset=FEATURES + [TARGET, TREATMENT])
# print(df.describe)
# ================================
# 4. Split Data
# ================================
df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
df_train = df_train.copy()
df_test = df_test.copy()

df_train_treat = df_train[df_train[TREATMENT] == 1]
df_train_control = df_train[df_train[TREATMENT] == 0]

df_test_treat = df_test[df_test[TREATMENT] == 1]
df_test_control = df_test[df_test[TREATMENT] == 0]

print("Treatment size:", len(df_train_treat))
print("Control size:", len(df_train_control))
# ================================
# 5. Train Models
# ================================

# --- Treatment Model ---
X_train_treat = df_train_treat[FEATURES]
y_train_treat = df_train_treat[TARGET]

model_treat = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

model_treat.fit(X_train_treat, y_train_treat)


# --- Control Model ---
X_train_control = df_train_control[FEATURES]
y_train_control = df_train_control[TARGET]

model_control = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

model_control.fit(X_train_control, y_train_control)
# ================================
# 6. Predict Probabilities
# ================================

X_test = df_test[FEATURES]

# Probability of booking WITH discount
p_treat = model_treat.predict_proba(X_test)[:, 1]
# print(p_treat)
# Probability of booking WITHOUT discount
p_control = model_control.predict_proba(X_test)[:, 1]

df_test["p_treat"] = p_treat
df_test["p_control"] = p_control
# ================================
# 7. Uplift Calculation
# ================================

df_test["uplift"] = df_test["p_treat"] - df_test["p_control"]

df_test[["p_treat", "p_control", "uplift"]].head()
# =================
# Evaluate model performance

y_test_treat = df_test_treat[TARGET]
X_test_treat = df_test_treat[FEATURES]
# On held-out treated users (if you reserve some)
print("AUC for Treatment Model:", roc_auc_score(y_test_treat,
      model_treat.predict_proba(X_test_treat)[:, 1]))
y_test_control = df_test_control[TARGET]
X_test_control = df_test_control[FEATURES]
print("AUC for Control Model:", roc_auc_score(y_test_control,
      model_control.predict_proba(X_test_control)[:, 1]))

# ================================
# 8. Uplift Segmentation
# ================================


def categorical_segment(row):
    if row['uplift'] > 0.08:  # Threshold for meaningful impact
        return 'Persuadable'
    elif row['uplift'] < -0.05:
        return 'Sleeping Dog'
    elif row['p_control'] > 0.7:
        return 'Sure Thing'
    else:
        return 'Lost Cause'


df_test["uplift_segment"] = df_test.apply(categorical_segment, axis=1)
# ================================
# 9. Compare Conversion by Segment
# ================================

segment_performance = df_test.groupby("uplift_segment")[TARGET].mean()

print(segment_performance)
# ================================
# 10. Uplift Distribution
# ================================

plt.figure()
plt.hist(df_test["uplift"], bins=50)
plt.title("Uplift Distribution")
plt.xlabel("Uplift")
plt.ylabel("Frequency")
plt.show()
# ================================
# 11. Targeting Strategy
# ================================

# Sort by uplift
df_sorted = df_test.sort_values(by="uplift", ascending=False)

# Top 20% users
top_20 = df_sorted.head(int(0.2 * len(df_test)))

print("Average uplift (top 20%):", top_20["uplift"].mean())
print("Conversion rate (top 20%):", top_20[TARGET].mean())
# ================================
# Uplift Curve (Incremental Gain)
# ================================

# Sort by predicted uplift (descending)
df_sorted = df_test.sort_values(
    by="uplift", ascending=False).reset_index(drop=True)

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
df_qini = df_test.sort_values(
    by="uplift", ascending=False).reset_index(drop=True)

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
overall_treat_rate = df_test[df_test[TREATMENT] == 1][TARGET].mean()
overall_control_rate = df_test[df_test[TREATMENT] == 0][TARGET].mean()

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

# Train on test data
model_base = RandomForestClassifier(
    n_estimators=100, max_depth=6, random_state=42)

model_base.fit(df_train[FEATURES], df_train[TARGET])

df_test["p_book"] = model_base.predict_proba(df_test[FEATURES])[:, 1]
df_test["p_treat"]     # booking with discount
df_test["p_control"]   # booking without discount
df_test["uplift"]      # difference

# ================================
# aggregate to user-level
# ================================
user_df = df_test.groupby('user_id').agg(
    p_treat=('p_treat', 'mean'),
    p_control=('p_control', 'mean'),
    uplift=('uplift', 'mean'),
    p_book=('p_book', 'mean'),
    hotel_booked=(TARGET, 'max'),       # did they ever book?
    hotel_discount=(TREATMENT, 'max'),  # did they ever get a discount?
).reset_index()


# ================================
# Decision Engine
# ================================
COST_DISCOUNT = 15
REVENUE_BOOKING = 100

user_df["expected_value"] = (
    user_df["uplift"] * REVENUE_BOOKING - COST_DISCOUNT
)


def assign_reward(row):
    # Already likely to book — don't waste the discount
    if row["p_book"] >= 0.7:
        return "No Discount (Sure Thing)"
    # Discount generates positive ROI for uncertain users
    elif row["expected_value"] > 0:
        return "Give Discount"
    # Discount actively hurts conversion
    elif row["uplift"] < 0:
        return "Avoid (Do Not Disturb)"
    else:
        return "Low Priority"


user_df["decision"] = user_df.apply(assign_reward, axis=1)
print(user_df["decision"].value_counts())
# Compare conversion rates by decision
print('Conversion rates by decision:')
print(user_df.groupby("decision")[TARGET].mean())

float_cols = user_df.select_dtypes(include='float').columns
user_df[float_cols] = user_df[float_cols].round(2)
user_df = pd.DataFrame(user_df[['user_id', 'decision']])
# Save the DataFrame to CSV-format file
output_path = (
    'C:/Users/soghr/TravelTide_Project/datasets/uplift_decision_per_user.csv'
)
user_df.to_csv(output_path, index=False)
