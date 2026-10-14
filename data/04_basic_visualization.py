from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Find project folders safely
# --------------------------------------------------

current_file = Path(__file__).resolve()

data_folder = current_file.parent
project_folder = data_folder.parent

data_path = data_folder / r"C:\Users\Ajay\Downloads\archive\Clean_Dataset.csv"
reports_folder = project_folder / "reports"

# Create reports folder if it does not exist
reports_folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. Chart 1: Flights by Airline
# --------------------------------------------------

airline_counts = df["airline"].value_counts()

plt.figure(figsize=(10, 6))
airline_counts.plot(kind="bar")

plt.title("Number of Flights by Airline")
plt.xlabel("Airline")
plt.ylabel("Number of Flights")
plt.xticks(rotation=45)
plt.tight_layout()

chart_path = reports_folder / "flights_by_airline.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()

print("Saved:", chart_path)


# --------------------------------------------------
# 4. Chart 2: Flights by Source City
# --------------------------------------------------

source_counts = df["source_city"].value_counts()

plt.figure(figsize=(10, 6))
source_counts.plot(kind="bar")

plt.title("Number of Flights by Source City")
plt.xlabel("Source City")
plt.ylabel("Number of Flights")
plt.xticks(rotation=45)
plt.tight_layout()

chart_path = reports_folder / "flights_by_source_city.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()

print("Saved:", chart_path)


# --------------------------------------------------
# 5. Chart 3: Flights by Class
# --------------------------------------------------

class_counts = df["class"].value_counts()

plt.figure(figsize=(8, 6))
class_counts.plot(kind="bar")

plt.title("Number of Flights by Class")
plt.xlabel("Class")
plt.ylabel("Number of Flights")
plt.xticks(rotation=0)
plt.tight_layout()

chart_path = reports_folder / "flights_by_class.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()

print("Saved:", chart_path)


# --------------------------------------------------
# 6. Chart 4: Price Distribution
# --------------------------------------------------

plt.figure(figsize=(10, 6))
df["price"].plot(kind="hist", bins=50)

plt.title("Flight Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.tight_layout()

chart_path = reports_folder / "price_distribution.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()

print("Saved:", chart_path)


# --------------------------------------------------
# 7. Final confirmation
# --------------------------------------------------

print("\nAll four charts were created successfully!")
print("\nExact reports folder:")
print(reports_folder)

print("\nFiles inside reports folder:")

for file in reports_folder.iterdir():
    print(file)