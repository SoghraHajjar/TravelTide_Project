import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv(
    "C:/Users/soghr/ML_Project/Mastery_project/data_cleaned.csv")

plt.figure(figsize=(10, 6))
# Drop NaNs before plotting the histogram to only show valid durations
data['trip_duration'].dropna().hist(bins=50, edgecolor='black')
plt.title('Distribution of Trip Duration (days)')
plt.xlabel('Trip Duration (days)')
plt.ylabel('Number of Flights')
plt.tight_layout()
plt.show()
