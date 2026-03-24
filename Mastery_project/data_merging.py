import pandas as pd
data_users = pd.read_csv(
    "C:/Users/soghr/TravelTide_Project/valid_users.csv")
data_sessions = pd.read_csv(
    "C:/Users/soghr/TravelTide_Project/Elena_suggested_sessions.csv")
data_flights = pd.read_csv(
    "C:/Users/soghr/TravelTide_Project/valid_flights.csv")
data_hotels = pd.read_csv(
    "C:/Users/soghr/TravelTide_Project/valid_hotels.csv")
####################
df_merged = pd.merge(data_sessions,
                     data_users, how="left", on=["user_id", "user_id"])
df_merged = pd.merge(df_merged,
                     data_flights, how="left", on=["trip_id", "trip_id"])
df_merged = pd.merge(df_merged,
                     data_hotels, how="left", on=["trip_id", "trip_id"])
