import pandas as pd
import functions
import data_prepration

data = pd.DataFrame(data_prepration.data)
# Calculate trip duration in days
cancelled_trip_ids = data[data['cancellation'] is True]['trip_id']

valid_bookings = data[(~data['trip_id'].isin(cancelled_trip_ids)) & (
    ~data['trip_id'].isna())].copy()

valid_bookings['trip_duration'] = (
    (valid_bookings['return_time'] -
     valid_bookings['departure_time']).dt.total_seconds()
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
trip_duration = valid_bookings.groupby('user_id')['trip_duration'].mean()
trip_duration.name = 'trip_duration'

cancellation_rate = data[data['trip_id'].notnull()].groupby('user_id')[
    'cancellation'].mean()
cancellation_rate.name = 'cancellation_rate'

overcarrier = valid_bookings.groupby('user_id')['bags_per_seat'].mean()
overcarrier.name = 'overcarrier'

discount = valid_bookings.groupby('user_id')['discount'].mean()
discount.name = 'discount'

customer_value = valid_bookings.groupby('user_id')['paied_amount_usd'].mean()
customer_value.name = 'customer_value'

user_features = pd.DataFrame({
    'user_id': data['user_id'].unique()
})

user_features = (user_features
                 .merge(cancellation_rate, on='user_id', how='left')
                 .merge(overcarrier, on='user_id', how='left')
                 .merge(trip_duration, on='user_id', how='left')
                 .merge(discount, on='user_id', how='left')
                 .merge(customer_value, on='user_id', how='left')
                 )

output_path = 'C:/Users/soghr/ML_Project/Mastery_project/user_features.csv'

# Save the DataFrame to CSV-format file
user_features.to_csv(output_path, index=False)
