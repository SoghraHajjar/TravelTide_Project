
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np

data_features = pd.DataFrame(pd.read_csv(
    'C:/Users/soghr/TravelTide_Project/datasets/user_features.csv'))

data = pd.DataFrame(pd.read_csv(
    "C:/Users/soghr/TravelTide_Project/datasets/merged_data.csv"))

data = data.merge(data_features[[
                  'user_id', 'Ave_monetary_value',
                  'session_per_user', 'Monetary_title']],
                  on='user_id', how='left')
data['flight_discount'] = np.where(data['flight_discount'] == 1, 1, 0)
data['hotel_discount'] = np.where(data['hotel_discount'] == 1, 1, 0)
data['married'] = np.where(data['married'] == True, 1, 0)
data['has_children'] = np.where(data['has_children'] == True, 1, 0)
data = data.fillna(0)
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
data['age_category'] = pd.cut(
    data['age'],
    bins=[0, 25, 60, data['age'].max()],
    labels=['Youth', 'Middle-age', 'Elderly'],
    right=True
)

X = data[[
    'page_clicks',
    'session_duration',
    'flight_discount',
    'hotel_discount',
    'age_category',
    'gender',
    'married',
    'has_children'
]]
# Create dummy variables for 'age_category' and concatenate them to X
age_dummies = pd.get_dummies(
    data['age_category'], prefix='age_category', drop_first=True, dtype=int)
# Ensure the dummy variables are integers
age_dummies = age_dummies.astype(int)
X = pd.concat([X, age_dummies], axis=1)
X = X.drop('age_category', axis=1)
# Create dummy variables for 'gender' and concatenate them to X
gender_dummies = pd.get_dummies(
    data['gender'], prefix='gender', drop_first=True, dtype=int)
# Ensure the dummy variables are integers
gender_dummies = gender_dummies.astype(int)
X = pd.concat([X, gender_dummies], axis=1)
X = X.drop('gender', axis=1)
X = X.fillna(0)
X = sm.add_constant(X)
y = np.where((data['flight_booked'] == 1) | (data['hotel_booked'] == 1), 1, 0)

model = sm.Logit(y, X).fit()
print(model.summary())

# Extract coefficients and confidence intervals
params = model.params
conf = model.conf_int()
conf.columns = ['lower', 'upper']
conf['coef'] = params

# Create a tidy DataFrame
coef_df = conf.reset_index()
coef_df.rename(columns={'index': 'variable'}, inplace=True)

# Plot
plt.figure(figsize=(10, 6))
sns.pointplot(
    data=coef_df,
    x='coef',
    y='variable',
    join=False,
    color='black'
)

# Add CI lines manually
for i, row in coef_df.iterrows():
    plt.plot([row['lower'], row['upper']], [i, i], color='blue')

plt.axvline(0, color='red', linestyle='--')
plt.title("Logistic Regression Coefficients with 95% Confidence Intervals")
plt.xlabel("Coefficient Estimate")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()
