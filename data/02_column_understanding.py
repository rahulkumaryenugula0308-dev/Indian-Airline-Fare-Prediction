import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\Ajay\Downloads\archive\Clean_Dataset.csv")

print("=" * 60)
print("COLUMN-WISE UNDERSTANDING")
print("=" * 60)

# Display unique values and number of unique values
for column in df.columns:
    print("\n" + "-" * 60)
    print("Column:", column)
    print("Data type:", df[column].dtype)
    print("Number of unique values:", df[column].nunique())

    # Display unique values only for columns with fewer than 20 categories
    if df[column].nunique() < 20:
        print("Unique values:", df[column].unique())