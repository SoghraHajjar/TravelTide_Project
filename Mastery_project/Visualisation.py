import matplotlib.pyplot as plt
import Feature_engineering
data = Feature_engineering.user_features

plt.figure(figsize=(10, 6))
# Drop NaNs before plotting the histogram to only show valid durations
data['trip_duration'].dropna().hist(bins=50, edgecolor='black')
plt.title('Distribution of Trip Duration (days)')
plt.xlabel('Trip Duration (days)')
plt.ylabel('Number of Flights')
plt.tight_layout()
plt.show()
