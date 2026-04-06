import pandas as pd
import numpy as np
import functions
user_features = pd.read_csv(
    "C:/Users/soghr/TravelTide_Project/valid_users.csv")

data = pd.DataFrame(pd.read_csv(
    "C:/Users/soghr/TravelTide_Project//Mastery_project/merged_data.csv"))

# changing format of date columns
date_columns = ['session_start', 'session_end', 'birthdate', 'sign_up_date',
                'departure_time', 'return_time', 'check_in_time',
                'check_out_time']
for col in date_columns:
    data[col] = pd.to_datetime(data[col], format='mixed', errors='coerce')
# changing format of boolian data
# data['return_flight_booked'] = data['return_flight_booked'].fillna(
#    False).astype(bool)
data['return_flight_booked'] = np.where(
    data['return_flight_booked'] == True, 1, 0)
# cancellation and booking
data['cancellation'] = np.where(data['cancellation'] == True, 1, 0)
data['hotel_booked'] = np.where(data['hotel_booked'] == True, 1, 0)
data['flight_booked'] = np.where(data['flight_booked'] == True, 1, 0)
data['booking_type'] = np.where(
    (data['hotel_booked'] == 1) & (data['flight_booked'] == 1), 'both',
    np.where(data['hotel_booked'] == 1, 'hotel',
             np.where(data['flight_booked'] == 1, 'flight', 'none')))

cancelled_trip_ids = data[data['cancellation'] == 1]['trip_id']
data['booking'] = np.where((~data['trip_id'].isin(cancelled_trip_ids)) & (
    ~data['trip_id'].isna()), 1, 0)

data['nights'] = np.where(data['nights'] == 0, 1, abs(data['nights']))
data['nights'] = data['nights'].fillna(0)
data['seats'] = data['seats'].fillna(0)

# defining valid bookings as those that are not cancelled and have a trip_id
valid_bookings = data[(~data['trip_id'].isin(cancelled_trip_ids)) & (
    ~data['trip_id'].isna())].copy()

# Calculate session duration in minutes
valid_bookings['session_duration'] = (
    (valid_bookings['session_end'] -
     valid_bookings['session_start']).dt.total_seconds()
    / 60
)
# time to flight
valid_bookings['time_to_flight'] = (
    (valid_bookings['departure_time'] -
     valid_bookings['session_end']).dt.total_seconds()
    / (24 * 3600)
)
# Calculation trip distance
valid_bookings['trip_distance'] = valid_bookings.apply(
    lambda row: functions.haversine(
        row['home_airport_lat'], row['home_airport_lon'],
        row['destination_airport_lat'], row['destination_airport_lon']
    ), axis=1
)
valid_bookings['trip_distance'].fillna(0, inplace=True)

# trip with discoun
valid_bookings['flight_discount'] = valid_bookings['flight_discount'].fillna(
    False).astype(bool)
valid_bookings['hotel_discount'] = valid_bookings['hotel_discount'].fillna(
    False).astype(bool)
valid_bookings['discount'] = (
    valid_bookings["flight_discount"] | valid_bookings["hotel_discount"]
)
valid_bookings['discount'].fillna(False, inplace=True)

# trip costs
valid_bookings['paid_flight'] = (
    valid_bookings['base_fare_usd'].fillna(0) *
    valid_bookings['seats'].fillna(0) *
    (1 - valid_bookings['flight_discount_amount'].fillna(0))
)
valid_bookings['paid_hotel'] = (
    valid_bookings['hotel_per_room_usd'].fillna(0) *
    valid_bookings['nights'].fillna(0) *
    valid_bookings['rooms'].fillna(0) *
    (1 - valid_bookings['hotel_discount_amount'].fillna(0))
)
valid_bookings['paied_amount_usd'] = valid_bookings['paid_flight'] + \
    valid_bookings['paid_hotel']

# print(valid_bookings[['paid_flight', 'paid_hotel',
#      'paied_amount_usd']].describe())
# overcarrying
valid_bookings['bags_per_seat'] = (
    valid_bookings['checked_bags'] /
    valid_bookings['seats'].replace(0, 1)
)

# per users
trip_seats = valid_bookings.groupby('user_id')['seats'].mean()
trip_seats.name = 'trip_seats'
ref_dict = {trip_seats.name: trip_seats}

trip_nights = valid_bookings.groupby('user_id')['nights'].mean()
trip_nights.name = 'trip_nights'
ref_dict['trip_nights'] = trip_nights

trip_distance = valid_bookings.groupby('user_id')['trip_distance'].mean()
trip_distance.name = 'trip_distance'
ref_dict['trip_distance'] = trip_distance

session_duration = valid_bookings.groupby('user_id')['session_duration'].mean()
session_duration.name = 'session_duration'

time_to_flight = valid_bookings.groupby(
    'user_id')['time_to_flight'].mean()
time_to_flight.name = 'time_to_flight'
ref_dict['time_to_flight'] = time_to_flight


cancellation_rate = data[data['trip_id'].notnull()].groupby('user_id')[
    'cancellation'].mean()
cancellation_rate.name = 'cancellation_rate'
ref_dict['cancellation_rate'] = cancellation_rate

n_booking = data[data['trip_id'].notnull()].groupby('user_id')[
    'booking'].count()
n_booking.name = 'n_booking'
ref_dict['n_booking'] = n_booking

n_hotel_booking = data[data['hotel_booked'] == 1].groupby('user_id')[
    'hotel_booked'].count()
n_hotel_booking.name = 'n_hotel_booking'
ref_dict['n_hotel_booking'] = n_hotel_booking

n_flight_booking = data[data['flight_booked'] == 1].groupby('user_id')[
    'flight_booked'].count()
n_flight_booking.name = 'n_flight_booking'
ref_dict['n_flight_booking'] = n_flight_booking

n_booking_types = data.groupby('user_id')[
    'booking_type'].nunique()
n_booking_types.name = 'n_booking_types'
ref_dict['n_booking_types'] = n_booking_types

session_per_user = data.groupby('user_id')[
    'session_id'].count()
session_per_user.name = 'session_per_user'
ref_dict['session_per_user'] = session_per_user

last_session = data[data['session_id'].notnull()].groupby('user_id')[
    'session_end'].max()
last_session.name = 'last_session'
ref_dict['last_session'] = last_session

n_hotel_discount = valid_bookings.groupby('user_id')['hotel_discount'].count()
n_hotel_discount.name = 'n_hotel_discount'
ref_dict['n_hotel_discount'] = n_hotel_discount

n_flight_discount = valid_bookings.groupby(
    'user_id')['flight_discount'].count()
n_flight_discount.name = 'n_flight_discount'
ref_dict['n_flight_discount'] = n_flight_discount

Ave_bags_per_seat = valid_bookings.groupby('user_id')['bags_per_seat'].mean()
Ave_bags_per_seat.name = 'Ave_bags_per_seat'
ref_dict['Ave_bags_per_seat'] = Ave_bags_per_seat


discount = valid_bookings.groupby('user_id')['discount'].mean()
discount.name = 'discount'
ref_dict['discount'] = discount

Monetary_value = valid_bookings.groupby('user_id')['paied_amount_usd'].sum()
Monetary_value.name = 'Monetary_value'
ref_dict['Monetary_value'] = Monetary_value

Ave_monetary_value = valid_bookings.groupby(
    'user_id')['paied_amount_usd'].mean()
Ave_monetary_value.name = 'Ave_monetary_value'
ref_dict['Ave_monetary_value'] = Ave_monetary_value


user_features = user_features.drop(
    columns=['home_airport_lat', 'home_airport_lon'])
# users age
user_features['age'] = (pd.to_datetime(
    '2023-01-05') - pd.to_datetime(user_features['birthdate'])).dt.days // 365
user_features['age_category'] = pd.cut(
    user_features['age'],
    bins=[0, 25, 60, user_features['age'].max()],
    labels=['Youth', 'Middle-age', 'Elderly'],
    right=True
)
user_features['married'] = np.where(
    user_features['married'] == 1.0, 'Married', 'Single')
user_features['has_children'] = np.where(
    user_features['has_children'] == 1.0, 'Yes', 'No')
user_features['gender'] = np.where(
    user_features['gender'] == 'M', 'Male', 'Female')
# merging all features to one dataframe
for ref_name, ref_series in ref_dict.items():
    user_features = user_features.merge(ref_series, on='user_id', how='left')

user_features[user_features.select_dtypes(include='number').columns] = \
    user_features[user_features.select_dtypes(
        include='number').columns].fillna(0)
user_features['Monetary_title'] = pd.qcut(
    user_features['Monetary_value'], 5,
    labels=['At Risk', 'Low Spenders', 'Medium Spenders', 'High Spenders', 'Champions'])
user_features = user_features.round(2)
print("Number of missing values in each column:")
print(user_features.isna().sum())
print("Final user_features DataFrame:")
print(user_features)
print("Columns in user_features DataFrame:")
print(user_features.columns.tolist())
print("Number of rows in user_features DataFrame:", len(user_features))
print("Number of unique users in user_features DataFrame:",
      user_features['user_id'].nunique())

output_path = (
    'C:/Users/soghr/TravelTide_Project/Mastery_project/user_features.csv'
)
output_path_2 = (
    'C:/Users/soghr/TravelTide_Project/Mastery_project/selected_user_features.csv'
)
# Save the DataFrame to CSV-format file
# selected_features = user_features[user_features['age',
#                                                'Monetary_title', 'Monetary_value', 'user_id', 'n_booking_types']]
user_features.to_csv(output_path, index=False)
