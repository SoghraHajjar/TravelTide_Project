import pandas as pd
import math
import data_prepration

data = pd.DataFrame(data_prepration.data)
# Calculate trip duration in days
data['trip_duration'] = (
    (data['return_time'] - data['departure_time']).dt.total_seconds()
    / (24 * 3600)
)

# Distance between home and destination


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Erdradius in km
    # Grad -> Radiant
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    distance = R * c
    return distance


data['distance_trip(km)'] = data.apply(
    lambda row: haversine(
        row['home_airport_lat'], row['home_airport_lon'],
        row['destination_airport_lat'], row['destination_airport_lon']
    ), axis=1
)

# is_school_holiday variable
# Initialize a set to store all school holidays
all_school_holidays = set()
# Get the unique years from the departure_time column
unique_years = data['departure_time'].dt.year.dropna(
).unique()  # only 2023
dates_spring = pd.date_range(start="2023-03-20", end="2023-04-04")
dates_winter = pd.date_range(start="2023-12-20", end="2024-01-04")
dates_summer = pd.date_range(start="2023-06-25", end="2024-09-04")

# Convert date ranges to sets of datetime.date objects.
for date in dates_spring:
    all_school_holidays.add(date.date())

for date in dates_winter:
    all_school_holidays.add(date.date())

for date in dates_summer:
    all_school_holidays.add(date.date())

# Re-create the binary variable 'is_school_holiday' with the updated set
data['is_school_holiday'] = data['departure_time'].dt.date.isin(
    all_school_holidays).astype(int)
