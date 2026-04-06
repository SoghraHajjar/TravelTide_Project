# from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns

data_features = pd.DataFrame(pd.read_csv(
    'C:/Users/soghr/TravelTide_Project/Mastery_project/user_features.csv'))

data = pd.DataFrame(pd.read_csv(
    "C:/Users/soghr/TravelTide_Project/Mastery_project/merged_data.csv"))

data = data.merge(data_features[[
                  'user_id', 'Ave_monetary_value',
                  'session_per_user', 'Monetary_title']],
                  on='user_id', how='left')

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
# data['discount'] = np.where((data['flight_discount'] == 1) | (
#    data['hotel_discount'] == 1), 1, 0)
data['age'] = (pd.to_datetime(
    '2023-01-05') - pd.to_datetime(data['birthdate'])).dt.days // 365
# print(data.columns)
# print(data_features.columns)
X = data[[
    'page_clicks',
    'session_duration',
    'session_per_user',
    'flight_discount',
    'hotel_discount',
    'home_country',
    'Ave_monetary_value',
    'age',
    'gender',
    'married',
    'has_children'
]]
# Create dummy variables for 'gender' and concatenate them to X
gender_dummies = pd.get_dummies(
    data['gender'], prefix='gender', drop_first=True)
X = pd.concat([X, gender_dummies], axis=1)
X = X.drop('gender', axis=1)

X['flight_discount'] = np.where(X['flight_discount'] == 1, 1, 0)
X['hotel_discount'] = np.where(X['hotel_discount'] == 1, 1, 0)
X['married'] = np.where(X['married'] == True, 1, 0)
X['has_children'] = np.where(X['has_children'] == True, 1, 0)
print(X.describe())
X = sm.add_constant(X)
y = np.where((data['flight_booked'] == 1) | (data['hotel_booked'] == 1), 1, 0)
#
# model = sm.Logit(y, X).fit()
# print(model.summary())
