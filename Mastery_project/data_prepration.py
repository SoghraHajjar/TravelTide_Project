import pandas as pd
import data_merging
data = pd.DataFrame(data_merging.df_merged)

# changing format of date columns
date_columns = ['session_start', 'session_end', 'birthdate', 'sign_up_date',
                'departure_time', 'return_time', 'check_in_time',
                'check_out_time']
for col in date_columns:
    data[col] = pd.to_datetime(data[col], format='mixed', errors='coerce')
# changing format of boolian data
data['return_flight_booked'] = data['return_flight_booked'].fillna(
    False).astype(bool)
###
hotel_city_name = data['hotel_name'].str.split('-', expand=True)
data['hotelname'] = hotel_city_name[0].str.strip()
data['hotel_city'] = hotel_city_name[1].str.strip()

output_path = 'C:/Users/soghr/ML_Project/Mastery_project/data_cleaned.csv'

# Save the DataFrame to CSV-format file
data.to_csv(output_path, index=False)
