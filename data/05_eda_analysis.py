from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Find project folders
# --------------------------------------------------

current_file = Path(__file__).resolve()

data_folder = current_file.parent
project_folder = data_folder.parent
reports_folder = project_folder / "reports"

reports_folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

data_path = data_folder / r"C:\Users\Ajay\Downloads\archive\Clean_Dataset.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. Average price by airline
# --------------------------------------------------

average_price = (
    df.groupby("airline")["price"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage price by airline:")
print(average_price)


# --------------------------------------------------
# 4. Create chart
# --------------------------------------------------

plt.figure(figsize=(10, 6))

average_price.plot(kind="bar")

plt.title("Average Flight Price by Airline")
plt.xlabel("Airline")
plt.ylabel("Average Price")
plt.xticks(rotation=45)
plt.tight_layout()


# --------------------------------------------------
# 5. Save chart
# --------------------------------------------------

chart_path = reports_folder / "average_price_by_airline.png"

plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()

print("\nChart saved successfully:")
print(chart_path)

# --------------------------------------------------
# 7. Average price by class
# --------------------------------------------------

average_price_by_class = (
    df.groupby("class")["price"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage price by class:")
print(average_price_by_class)


# --------------------------------------------------
# 8. Create class price chart
# --------------------------------------------------

plt.figure(figsize=(8, 6))

average_price_by_class.plot(kind="bar")

plt.title("Average Flight Price by Class")
plt.xlabel("Class")
plt.ylabel("Average Price")
plt.xticks(rotation=0)
plt.tight_layout()


# --------------------------------------------------
# 9. Save chart
# --------------------------------------------------

chart_path = reports_folder / "average_price_by_class.png"

plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()

print("\nClass price chart saved successfully:")
print(chart_path)
# --------------------------------------------------
# 10. Average price by days left
# --------------------------------------------------

average_price_by_days = (
    df.groupby("days_left")["price"]
    .mean()
)

print("\nAverage price by days left:")
print(average_price_by_days)


# --------------------------------------------------
# 11. Create days-left price chart
# --------------------------------------------------

plt.figure(figsize=(10, 6))

average_price_by_days.plot(kind="line", marker="o")

plt.title("Average Flight Price by Days Left")
plt.xlabel("Days Left Before Flight")
plt.ylabel("Average Price")
plt.grid(True)
plt.tight_layout()


# --------------------------------------------------
# 12. Save chart
# --------------------------------------------------

chart_path = reports_folder / "average_price_by_days_left.png"

plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()

print("\nDays-left chart saved successfully:")
print(chart_path)
# --------------------------------------------------
# 13. Average price by number of stops
# --------------------------------------------------

average_price_by_stops = (
    df.groupby("stops")["price"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage price by stops:")
print(average_price_by_stops)


# --------------------------------------------------
# 14. Create stops price chart
# --------------------------------------------------

plt.figure(figsize=(8, 6))

average_price_by_stops.plot(kind="bar")

plt.title("Average Flight Price by Number of Stops")
plt.xlabel("Stops")
plt.ylabel("Average Price")
plt.xticks(rotation=0)
plt.tight_layout()


# --------------------------------------------------
# 15. Save chart
# --------------------------------------------------

chart_path = reports_folder / "average_price_by_stops.png"

plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()

print("\nStops price chart saved successfully:")
print(chart_path)
# --------------------------------------------------
# 16. Average price by route
# --------------------------------------------------

average_price_by_route = (
    df.groupby(["source_city", "destination_city"])["price"]
    .mean()
    .sort_values(ascending=False)
)

print("\nTop 10 most expensive routes:")

print(average_price_by_route.head(10))


# --------------------------------------------------
# 17. Prepare route names
# --------------------------------------------------

route_prices = average_price_by_route.head(10).sort_values()

route_names = [
    f"{source} → {destination}"
    for source, destination in route_prices.index
]


# --------------------------------------------------
# 18. Create route price chart
# --------------------------------------------------

plt.figure(figsize=(12, 7))

plt.barh(route_names, route_prices.values)

plt.title("Top 10 Most Expensive Routes")
plt.xlabel("Average Price")
plt.ylabel("Route")
plt.tight_layout()


# --------------------------------------------------
# 19. Save chart
# --------------------------------------------------

chart_path = reports_folder / "top_10_expensive_routes.png"

plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()

print("\nRoute price chart saved successfully:")
print(chart_path)
# --------------------------------------------------
# 20. Correlation analysis
# --------------------------------------------------

numeric_columns = [
    "duration",
    "days_left",
    "price"
]

correlation_matrix = df[numeric_columns].corr()

print("\nCorrelation matrix:")
print(correlation_matrix)


# --------------------------------------------------
# 21. Create correlation chart
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.imshow(correlation_matrix, cmap="coolwarm")

plt.colorbar()

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Correlation Matrix")

# Display correlation values inside the chart
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(
            j,
            i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.tight_layout()


# --------------------------------------------------
# 22. Save correlation chart
# --------------------------------------------------

chart_path = reports_folder / "correlation_matrix.png"

plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()

print("\nCorrelation chart saved successfully:")
print(chart_path)