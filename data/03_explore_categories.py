import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\Ajay\Downloads\archive\Clean_Dataset.csv")

print("=" * 60)
print("DATASET CATEGORY EXPLORATION")
print("=" * 60)

# 1. Airlines
print("\nAirlines:")
print(df["airline"].value_counts())

# 2. Source cities
print("\nSource cities:")
print(df["source_city"].value_counts())

# 3. Destination cities
print("\nDestination cities:")
print(df["destination_city"].value_counts())

# 4. Departure time categories
print("\nDeparture time categories:")
print(df["departure_time"].value_counts())

# 5. Arrival time categories
print("\nArrival time categories:")
print(df["arrival_time"].value_counts())

# 6. Stops
print("\nStops:")
print(df["stops"].value_counts())

# 7. Class
print("\nClass:")
print(df["class"].value_counts())

# 8. Flight numbers
print("\nNumber of unique flight numbers:")
print(df["flight"].nunique())

# 9. Duration summary
print("\nDuration summary:")
print(df["duration"].describe())

# 10. Days left summary
print("\nDays left summary:")
print(df["days_left"].describe())

# 11. Price summary
print("\nPrice summary:")
print(df["price"].describe())