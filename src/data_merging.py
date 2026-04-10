# importing libraries
import pandas as pd
# importing dataframes
data_users = pd.read_csv(
    "C:/Users/soghr/TravelTide_Project/datasets/valid_users.csv")
data_sessions = pd.read_csv(
    "C:/Users/soghr/TravelTide_Project/datasets/Elena_suggested_sessions.csv")
data_flights = pd.read_csv(
    "C:/Users/soghr/TravelTide_Project/datasets/valid_flights.csv")
data_hotels = pd.read_csv(
    "C:/Users/soghr/TravelTide_Project/datasets/valid_hotels.csv")
# ==================

# merging all dataframes to one dataframe
df_merged = pd.merge(data_sessions,
                     data_users, how="left", on=["user_id", "user_id"])
df_merged = pd.merge(df_merged,
                     data_flights, how="left", on=["trip_id", "trip_id"])
df_merged = pd.merge(df_merged,
                     data_hotels, how="left", on=["trip_id", "trip_id"])

# =============
hotel_city_name = df_merged['hotel_name'].str.split('-', expand=True)
df_merged['hotelname'] = hotel_city_name[0].str.strip()
df_merged['hotel_city'] = hotel_city_name[1].str.strip()


# Save the DataFrame to CSV-format file
output_path = (
    'C:/Users/soghr/TravelTide_Project/datasets/merged_data.csv'
)
df_merged.to_csv(output_path, index=False)
print("Data merging completed and saved to 'merged_data.csv'.")
