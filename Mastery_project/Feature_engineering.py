import pandas as pd
import numpy as np
import functions
import data_merging
import data_prepration

data = pd.DataFrame(data_prepration.data)

data['cancellation'] = np.where(data['cancellation'] == True, 1, 0)
cancelled_trip_ids = data[data['cancellation'] == 1]['trip_id']
data['booking'] = np.where((~data['trip_id'].isin(cancelled_trip_ids)) & (
    ~data['trip_id'].isna()), 1, 0)

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

# trip with discoun
valid_bookings['flight_discount'] = valid_bookings['flight_discount'].fillna(
    False).astype(bool)
valid_bookings['hotel_discount'] = valid_bookings['hotel_discount'].fillna(
    False).astype(bool)
valid_bookings['discount'] = (
    valid_bookings["flight_discount"] | valid_bookings["hotel_discount"]
)

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
# overcarrying
valid_bookings['bags_per_seat'] = valid_bookings['checked_bags'] / \
    valid_bookings['seats'].replace(0, 1)

# per users
# trip_bookings_rate = valid_bookings.groupby('user_id')['trip_id'].count()
# trip_bookings_rate.name = 'trip_bookings_rate'

trip_seats = valid_bookings.groupby('user_id')['seats'].mean()
trip_seats.name = 'trip_seats'

trip_distance = valid_bookings.groupby('user_id')['trip_distance'].mean()
trip_distance.name = 'trip_distance'

session_duration = valid_bookings.groupby('user_id')['session_duration'].mean()
session_duration.name = 'session_duration'

time_to_flight = valid_bookings.groupby(
    'user_id')['time_to_flight'].mean()
time_to_flight.name = 'time_to_flight'

cancellation_rate = data[data['trip_id'].notnull()].groupby('user_id')[
    'cancellation'].mean()
cancellation_rate.name = 'cancellation_rate'

n_booking = data[data['trip_id'].notnull()].groupby('user_id')[
    'booking'].count()
n_booking.name = 'n_booking'

session_per_user = data[data['session_id'].notnull()].groupby('user_id')[
    'session_id'].count()
session_per_user.name = 'session_per_user'

n_hotel_discount = valid_bookings.groupby('user_id')['hotel_discount'].count()
n_hotel_discount.name = 'n_hotel_discount'

n_flight_discount = valid_bookings.groupby(
    'user_id')['flight_discount'].count()
n_flight_discount.name = 'n_flight_discount'

overcarrier = valid_bookings.groupby('user_id')['bags_per_seat'].mean()
overcarrier.name = 'overcarrier'

discount = valid_bookings.groupby('user_id')['discount'].mean()
discount.name = 'discount'

customer_value = valid_bookings.groupby('user_id')['paied_amount_usd'].mean()
customer_value.name = 'customer_value'

user_features = data_merging.data_users
# users age
user_features['age'] = (pd.to_datetime(
    '2023-01-05') - pd.to_datetime(user_features['birthdate'])).dt.days // 365
# merging all features to one dataframe
user_features = (user_features
                 .merge(cancellation_rate, on='user_id', how='left')
                 .merge(n_hotel_discount, on='user_id', how='left')
                 .merge(n_flight_discount, on='user_id', how='left')
                 .merge(session_per_user, on='user_id', how='left')
                 .merge(n_booking, on='user_id', how='left')
                 .merge(trip_distance, on='user_id', how='left')
                 .merge(trip_seats, on='user_id', how='left')
                 .merge(overcarrier, on='user_id', how='left')
                 .merge(session_duration, on='user_id', how='left')
                 .merge(time_to_flight, on='user_id', how='left')
                 .merge(discount, on='user_id', how='left')
                 .merge(customer_value, on='user_id', how='left')
                 )

output_path = 'C:/Users/soghr/TravelTide_Project/Mastery_project/user_features.csv'

# Save the DataFrame to CSV-format file
user_features.to_csv(output_path, index=False)
