import pandas as pd
import numpy as np
# import functions

data_features = pd.DataFrame(pd.read_csv(
    'C:/Users/soghr/TravelTide_Project/Mastery_project/user_features.csv'))

data = pd.DataFrame(pd.read_csv(
    "C:/Users/soghr/TravelTide_Project/Mastery_project/merged_data.csv"))

# sessions_df is your sessions table loaded as a pandas DataFrame

# 1. Create helper columns
data["discount_shown"] = (
    (data["flight_discount"] == 1) |
    (data["hotel_discount"] == 1)
)

data["booking_made"] = (
    (data["flight_booked"] == 1) |
    (data["hotel_booked"] == 1)
)

# 2. Group by user and compute probabilities
user_stats = data.groupby("user_id").apply(
    lambda x: pd.Series({
        "discount_sessions": x["discount_shown"].sum(),
        "discount_bookings": (x["discount_shown"] & x["booking_made"]).sum(),
        "no_discount_sessions": (~x["discount_shown"]).sum(),
        "no_discount_bookings": ((~x["discount_shown"]) & x["booking_made"]).sum()
    })
).reset_index()

# 3. Calculate probabilities
user_stats["P(booking|discount)"] = user_stats["discount_bookings"] / \
    user_stats["discount_sessions"].replace(0, np.nan)
user_stats["P(booking|no_discount)"] = user_stats["no_discount_bookings"] / \
    user_stats["no_discount_sessions"].replace(0, np.nan)
user_stats["discount_effectiveness"] = user_stats["P(booking|discount)"] - \
    user_stats["P(booking|no_discount)"]
# 4. Replace infinities or NaN (e.g., users who never saw a discount)
user_stats = user_stats.fillna(0)
# other features
data_features = data_features[data_features['Monetary_value'] > 0]

data_features["p_booking"] = data_features["n_booking"] / \
    data_features["session_per_user"]

data_features["p_hotel_booking"] = data_features["n_hotel_booking"] / \
    data_features["n_booking"]
data_features["p_flight_booking"] = data_features["n_flight_booking"] / \
    data_features["n_booking"]

data_features["time_to_flight"] = data_features["time_to_flight"].fillna(0)
data_features["cancel_urgency"] = 1 / (1 + data_features["time_to_flight"])


perk_signals = (
    user_stats
    .merge(
        data_features[
            [
                "user_id",
                "age_category",
                "Monetary_title",
                "cancellation_rate",
                "Ave_bags_per_seat",
                "trip_nights",
                "p_booking",
                "p_hotel_booking",
                "p_flight_booking",
                "cancel_urgency"
            ]
        ],
        on="user_id",
        how="left"
    )
)
# perk_signals = perk_signals.fillna(0)
# 5. Normalize scores for each perk
de_min = perk_signals['discount_effectiveness'].min()
de_max = perk_signals['discount_effectiveness'].max()
perk_signals['discount_effectiveness'] = (
    (perk_signals['discount_effectiveness'] - de_min) / (de_max - de_min))
bags_min = perk_signals['Ave_bags_per_seat'].min()
bags_max = perk_signals['Ave_bags_per_seat'].max()
perk_signals['ave_bags_normalized'] = (
    (perk_signals['Ave_bags_per_seat'] - bags_min) / (bags_max - bags_min))
nights_min = perk_signals['trip_nights'].min()
nights_max = perk_signals['trip_nights'].max()
perk_signals['trip_nights_normalized'] = (
    (perk_signals['trip_nights'] - nights_min) / (nights_max - nights_min))

perk_signals['score_free_cancellation'] = 0.7*perk_signals['cancel_urgency']
+0.3*perk_signals['cancellation_rate']


perk_signals['score_free_checkedbag'] = 0.8*perk_signals['ave_bags_normalized'] + \
    0.2 * perk_signals['p_flight_booking']
perk_signals['score_exclusive_discount'] = perk_signals['discount_effectiveness']

perk_signals['score_free_meal'] = 0.7*perk_signals['trip_nights_normalized'] + \
    0.3 * perk_signals['p_hotel_booking']

perk_signals['score_free_night_flight'] = 0.7 * perk_signals['p_booking'] + \
    0.3 * perk_signals['trip_nights_normalized']

perk_signals = perk_signals.fillna(0)

# Combine into a single score matrix
score_cols = [
    "score_free_cancellation",
    "score_free_checkedbag",
    "score_exclusive_discount",
    "score_free_meal",
    "score_free_night_flight"
]

# 6. Determine top perk
perk_signals["top_perk"] = perk_signals[score_cols].idxmax(axis=1)
print(perk_signals.shape)
# print(data_features["cancellation_rate"].describe())
# print(data_features['n_booking'].describe())
output_path = (
    'C:/Users/soghr/TravelTide_Project/Mastery_project/perk_signals.csv'
)

# Save the DataFrame to CSV-format file
perk_signals = pd.DataFrame(perk_signals)
perk_signals.to_csv(output_path, encoding="utf-8", index=False)
