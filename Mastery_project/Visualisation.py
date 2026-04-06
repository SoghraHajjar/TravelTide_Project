import pandas as pd
import Feature_engineering
import matplotlib.pyplot as plt
import seaborn as sns
data = Feature_engineering.user_features

# Age analysis
plt.figure(figsize=(10, 6))
# Drop NaNs before plotting the histogram to only show valid durations
data['age'].dropna().hist(bins=50, edgecolor='black')
plt.title('Distribution of Age')
plt.xlabel('Age(year)')
plt.ylabel('Number of Flights')
plt.tight_layout()
plt.show()
# Categorizing Age into Groups
# Youth: Ages 0-25
# Middle-age: Ages 26-65
# Elderly: Ages 66 and above
bins = [0, 25, 65, data['age'].max()]
labels = ['Youth', 'Middle-age', 'Elderly']
data['age_category'] = pd.cut(
    data['age'], bins=bins, labels=labels, right=True)
print("Value counts for the new 'age_category' column:")
print(data['age_category'].value_counts())

# short flight time analysis
data['short_flight_time'] = data['time_to_flight'] < 5
print("Value counts for the new 'short_flight_time' column:")
print(data['short_flight_time'].value_counts(dropna=False))
plt.figure(figsize=(10, 6))
ax = sns.countplot(
    x='age_category',
    hue='short_flight_time',
    data=data.dropna(subset=['age_category', 'short_flight_time']),
    palette='viridis',
    order=labels
)

plt.title('Age Categories by Short Flight Time (Percentages)')
plt.xlabel('Age Category')
plt.ylabel('Number of Users')
plt.legend(title='Short Flight (< 5 days)')

# Calculate percentages and add annotations
total_counts_by_age = data.dropna(
    subset=['age_category', 'short_flight_time']
).groupby('age_category').size()

for p in ax.patches:
    height = p.get_height()
    # Get the x-coordinate of the center of the bar
    age_category = p.get_x() + p.get_width() / 2

    # Determine the exact age category from the order
    # (Youth, Middle-age, Elderly)
    category_index = int(p.get_x() + 0.5)
    current_age_category_label = labels[category_index]

    total_in_category = total_counts_by_age.get(
        current_age_category_label, 0)
    if total_in_category > 0:
        percentage = 100 * height / total_in_category
        ax.annotate(f'{percentage:.1f}%',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9,
                    color='black', xytext=(0, 5),
                    textcoords='offset points')
plt.tight_layout()
plt.show()
# trip discount analysis
plt.figure(figsize=(10, 6))
sns.boxplot(x='discount', y='age_category', data=data.dropna(
    subset=['discount', 'age_category']), palette='viridis', order=labels)
plt.title('Distribution of Discount by Age Category')
plt.xlabel('Discount')
plt.ylabel('Age Category')
plt.tight_layout()
plt.show()

# Trip duration analysis
plt.figure(figsize=(10, 6))
data['trip_duration'].dropna().hist(bins=50, edgecolor='black')
plt.title('Distribution of Trip Duration (days)')
plt.xlabel('Trip Duration (days)')
plt.ylabel('Number of Flights')
plt.tight_layout()
plt.show()

# Create the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['age'], y=data['trip_distance'], data=data)
plt.title('Age vs. Trip Distance')
plt.xlabel('Age')
plt.ylabel('Trip Distance (km)')
plt.grid(True)
plt.show()
