
# %%
# --------------------------------------------------
# 1. Find project folders and load libraries
# --------------------------------------------------

from pathlib import Path

import pandas as pd


current_file = Path(__file__).resolve()

data_folder = current_file.parent
project_folder = data_folder.parent

data_path = data_folder / r"C:\Users\Ajay\Downloads\archive\Clean_Dataset.csv"


# %%
# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)

print("\nOriginal columns:")
print(df.columns.tolist())


# %%
# --------------------------------------------------
# 3. Remove unnecessary index column
# --------------------------------------------------

df = df.drop(columns=["Unnamed: 0"])


# %%
# --------------------------------------------------
# 4. Display updated dataset information
# --------------------------------------------------

print("\nUpdated shape:", df.shape)

print("\nUpdated columns:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())


# %%
# --------------------------------------------------
# 5. Create route feature
# --------------------------------------------------

df["route"] = (
    df["source_city"] + "_" + df["destination_city"]
)


# %%
# --------------------------------------------------
# 6. Display route information
# --------------------------------------------------

print("\nRoute feature created successfully!")

print("\nFirst five routes:")

print(
    df[
        [
            "source_city",
            "destination_city",
            "route"
        ]
    ].head()
)


# %%
# --------------------------------------------------
# 7. Count unique routes
# --------------------------------------------------

print("\nNumber of unique routes:")

print(df["route"].nunique())


# %%
# --------------------------------------------------
# 8. Display most common routes
# --------------------------------------------------

print("\nTop 10 most common routes:")

print(df["route"].value_counts().head(10))


# %%
# --------------------------------------------------
# 9. Separate features and target
# --------------------------------------------------

# We remove:
# 1. price  -> target variable
# 2. flight -> high-cardinality feature
#
# flight has many unique values, so removing it
# reduces the number of encoded features.

X = df.drop(columns=["price", "flight"])

y = df["price"]


# %%
# --------------------------------------------------
# 10. Display feature and target information
# --------------------------------------------------

print("\nFeatures X shape:")
print(X.shape)

print("\nTarget y shape:")
print(y.shape)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget column:")
print(y.name)

print("\nFirst five feature rows:")
print(X.head())

print("\nFirst five target values:")
print(y.head())


# %%
# --------------------------------------------------
# 11. Identify categorical and numerical columns
# --------------------------------------------------

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# %%
# --------------------------------------------------
# 12. Display column types
# --------------------------------------------------

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)

print("\nNumber of categorical columns:")
print(len(categorical_columns))

print("\nNumber of numerical columns:")
print(len(numerical_columns))


# %%
# --------------------------------------------------
# 13. Train-Test Split
# --------------------------------------------------

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTrain-Test Split Completed")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# %%
# --------------------------------------------------
# 14. Data Preprocessing
# --------------------------------------------------

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        )
    ]
)


print("\nPreprocessor created successfully")

print("Categorical columns:", categorical_columns)

print("Numerical columns:", numerical_columns)


# %%
# --------------------------------------------------
# 15. Fit and Transform Data
# --------------------------------------------------

# Fit the preprocessor only on training data

X_train_transformed = preprocessor.fit_transform(
    X_train
)


# Transform test data using the same preprocessor

X_test_transformed = preprocessor.transform(
    X_test
)


print("\nData Transformation Completed")

print(
    "X_train_transformed shape:",
    X_train_transformed.shape
)

print(
    "X_test_transformed shape:",
    X_test_transformed.shape
)


# %%
# --------------------------------------------------
# 16. Train Linear Regression Model
# --------------------------------------------------

from sklearn.linear_model import LinearRegression


model = LinearRegression()


print("\nTraining Linear Regression model...")


model.fit(
    X_train_transformed,
    y_train
)


print("Linear Regression model trained successfully")


# %%
# --------------------------------------------------
# 17. Make Linear Regression Predictions
# --------------------------------------------------

print("\nMaking predictions...")


y_pred = model.predict(
    X_test_transformed
)


print("Predictions completed successfully")


print("\nFirst 10 actual prices:")

print(
    y_test.head(10).values
)


print("\nFirst 10 predicted prices:")

print(
    y_pred[:10]
)


# %%
# --------------------------------------------------
# 18. Linear Regression Model Evaluation
# --------------------------------------------------

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np


mae = mean_absolute_error(
    y_test,
    y_pred
)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)


r2 = r2_score(
    y_test,
    y_pred
)


print("\nModel Evaluation Results")

print("------------------------")

print("MAE :", mae)

print("RMSE:", rmse)

print("R2  :", r2)


# %%
# --------------------------------------------------
# 19. Random Forest Regressor
# --------------------------------------------------

from sklearn.ensemble import RandomForestRegressor


rf_model = RandomForestRegressor(
    n_estimators=30,
    random_state=42,
    n_jobs=-1
)


print("\nTraining Random Forest model...")


rf_model.fit(
    X_train_transformed,
    y_train
)


print("Random Forest model trained successfully")


# %%
# --------------------------------------------------
# 20. Random Forest Predictions
# --------------------------------------------------

print("\nMaking Random Forest predictions...")


rf_pred = rf_model.predict(
    X_test_transformed
)


print(
    "Random Forest predictions completed successfully"
)


print("\nFirst 10 actual prices:")

print(
    y_test.head(10).values
)


print("\nFirst 10 Random Forest predicted prices:")

print(
    rf_pred[:10]
)


# %%
# --------------------------------------------------
# 21. Random Forest Evaluation
# --------------------------------------------------

rf_mae = mean_absolute_error(
    y_test,
    rf_pred
)


rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_pred
    )
)


rf_r2 = r2_score(
    y_test,
    rf_pred
)


print("\nRandom Forest Evaluation Results")

print("--------------------------------")

print("MAE :", rf_mae)

print("RMSE:", rf_rmse)

print("R2  :", rf_r2)



# %%
# --------------------------------------------------
# 22. Random Forest Overfitting Check
# --------------------------------------------------

rf_train_pred = rf_model.predict(X_train_transformed)

train_mae = mean_absolute_error(
    y_train,
    rf_train_pred
)

train_rmse = np.sqrt(
    mean_squared_error(
        y_train,
        rf_train_pred
    )
)

train_r2 = r2_score(
    y_train,
    rf_train_pred
)


print("\nRandom Forest Training Performance")
print("----------------------------------")

print("Training MAE :", train_mae)
print("Training RMSE:", train_rmse)
print("Training R2  :", train_r2)


print("\nRandom Forest Testing Performance")
print("---------------------------------")

print("Testing MAE :", rf_mae)
print("Testing RMSE:", rf_rmse)
print("Testing R2  :", rf_r2)


# %%
# --------------------------------------------------
# 24. Decision Tree Regressor
# --------------------------------------------------

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Create model
dt_model = DecisionTreeRegressor(
    random_state=42
)

# Train model
dt_model.fit(X_train_transformed, y_train)

# Predictions
dt_train_pred = dt_model.predict(X_train_transformed)
dt_test_pred = dt_model.predict(X_test_transformed)

# Training metrics
dt_train_mae = mean_absolute_error(y_train, dt_train_pred)
dt_train_rmse = np.sqrt(mean_squared_error(y_train, dt_train_pred))
dt_train_r2 = r2_score(y_train, dt_train_pred)

# Testing metrics
dt_test_mae = mean_absolute_error(y_test, dt_test_pred)
dt_test_rmse = np.sqrt(mean_squared_error(y_test, dt_test_pred))
dt_test_r2 = r2_score(y_test, dt_test_pred)

# Display results
print("\nDecision Tree Regressor")
print("----------------------")

print("\nTraining Performance")
print("MAE :", dt_train_mae)
print("RMSE:", dt_train_rmse)
print("R2  :", dt_train_r2)

print("\nTesting Performance")
print("MAE :", dt_test_mae)
print("RMSE:", dt_test_rmse)
print("R2  :", dt_test_r2)

# %%
# --------------------------------------------------
# 25. Gradient Boosting Regressor
# --------------------------------------------------

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Create model
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Train model
gb_model.fit(X_train_transformed, y_train)

# Predictions
gb_train_pred = gb_model.predict(X_train_transformed)
gb_test_pred = gb_model.predict(X_test_transformed)

# Training metrics
gb_train_mae = mean_absolute_error(y_train, gb_train_pred)
gb_train_rmse = np.sqrt(mean_squared_error(y_train, gb_train_pred))
gb_train_r2 = r2_score(y_train, gb_train_pred)

# Testing metrics
gb_test_mae = mean_absolute_error(y_test, gb_test_pred)
gb_test_rmse = np.sqrt(mean_squared_error(y_test, gb_test_pred))
gb_test_r2 = r2_score(y_test, gb_test_pred)

# Display results
print("\nGradient Boosting Regressor")
print("--------------------------")

print("\nTraining Performance")
print("MAE :", gb_train_mae)
print("RMSE:", gb_train_rmse)
print("R2  :", gb_train_r2)

print("\nTesting Performance")
print("MAE :", gb_test_mae)
print("RMSE:", gb_test_rmse)
print("R2  :", gb_test_r2)


# %%
# --------------------------------------------------
# 26. HistGradientBoosting Regressor
# --------------------------------------------------

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Convert sparse data to dense
X_train_dense = X_train_transformed.toarray()
X_test_dense = X_test_transformed.toarray()

print("Dense training shape:", X_train_dense.shape)
print("Dense testing shape :", X_test_dense.shape)

# Create model
hgb_model = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=42
)

# Train model
hgb_model.fit(X_train_dense, y_train)

# Predictions
hgb_train_pred = hgb_model.predict(X_train_dense)
hgb_test_pred = hgb_model.predict(X_test_dense)

# Training metrics
hgb_train_mae = mean_absolute_error(y_train, hgb_train_pred)
hgb_train_rmse = np.sqrt(mean_squared_error(y_train, hgb_train_pred))
hgb_train_r2 = r2_score(y_train, hgb_train_pred)

# Testing metrics
hgb_test_mae = mean_absolute_error(y_test, hgb_test_pred)
hgb_test_rmse = np.sqrt(mean_squared_error(y_test, hgb_test_pred))
hgb_test_r2 = r2_score(y_test, hgb_test_pred)

# Display results
print("\nHistGradientBoosting Regressor")
print("-----------------------------")

print("\nTraining Performance")
print("MAE :", hgb_train_mae)
print("RMSE:", hgb_train_rmse)
print("R2  :", hgb_train_r2)

print("\nTesting Performance")
print("MAE :", hgb_test_mae)
print("RMSE:", hgb_test_rmse)
print("R2  :", hgb_test_r2)



# %%
# --------------------------------------------------
# 27. Linear Regression Training Performance
# --------------------------------------------------

# Training predictions
lr_train_pred = model.predict(X_train_transformed)

# Training metrics
lr_train_mae = mean_absolute_error(y_train, lr_train_pred)
lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr_train_pred))
lr_train_r2 = r2_score(y_train, lr_train_pred)

print("\nLinear Regression Training Performance")
print("--------------------------------------")

print("MAE :", lr_train_mae)
print("RMSE:", lr_train_rmse)
print("R2  :", lr_train_r2)


# %% Final Model Comparison

import pandas as pd

model_comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "HistGradientBoosting"
    ],

    "Train MAE": [
        4572.60,
        58.31,
        425.08,
        2877.91,
        2252.65
    ],

    "Test MAE": [
        4243.64,
        1146.10,
        1084.57,
        2878.23,
        2278.97
    ],

    "Train RMSE": [
        6719.70,
        606.20,
        1164.27,
        4855.99,
        3823.63
    ],

    "Test RMSE": [
        6180.08,
        3451.01,
        2808.00,
        4900.97,
        3908.79
    ],

    "Train R2": [
        0.9123,
        0.9993,
        0.9974,
        0.9542,
        0.9716
    ],

    "Test R2": [
        0.9259,
        0.9769,
        0.9847,
        0.9534,
        0.9704
    ]
})

# Calculate train-test R² gap
model_comparison["R2 Gap"] = (
    model_comparison["Train R2"] -
    model_comparison["Test R2"]
)

print(model_comparison.to_string(index=False))

# %% Random Forest Hyperparameter Tuning

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

rf_configs = [
    {
        "n_estimators": 30,
        "max_depth": None,
        "min_samples_leaf": 1
    },
    {
        "n_estimators": 30,
        "max_depth": 20,
        "min_samples_leaf": 1
    },
    {
        "n_estimators": 30,
        "max_depth": 25,
        "min_samples_leaf": 1
    },
    {
        "n_estimators": 30,
        "max_depth": 30,
        "min_samples_leaf": 1
    }
]

results = []

for config in rf_configs:

    print("Training:", config)

    rf = RandomForestRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train_transformed, y_train)

    train_pred = rf.predict(X_train_transformed)
    test_pred = rf.predict(X_test_transformed)

    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    results.append({
        "max_depth": config["max_depth"],
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Train R2": train_r2,
        "Test R2": test_r2
    })

rf_tuning_results = pd.DataFrame(results)

print("\nRandom Forest Tuning Results:")
print(rf_tuning_results.to_string(index=False))

# %% Save Final Model and Preprocessor

import joblib
from pathlib import Path

# Create models folder
project_root = Path(__file__).resolve().parent.parent
models_folder = project_root / "models"
models_folder.mkdir(exist_ok=True)

# Train final Random Forest model
final_rf_model = RandomForestRegressor(
    n_estimators=30,
    random_state=42,
    n_jobs=-1
)

final_rf_model.fit(X_train_transformed, y_train)

# Save model
model_path = models_folder / "random_forest_fare_model.pkl"
joblib.dump(final_rf_model, model_path)

# Save preprocessor
preprocessor_path = models_folder / "fare_preprocessor.pkl"
joblib.dump(preprocessor, preprocessor_path)

print("Final Random Forest model saved successfully!")
print("Model path:", model_path)

print("\nPreprocessor saved successfully!")
print("Preprocessor path:", preprocessor_path)


# %% Test Fare Prediction

import pandas as pd
import joblib
from pathlib import Path

# Load saved model and preprocessor
project_root = Path(__file__).resolve().parent.parent
models_folder = project_root / "models"

model_path = models_folder / "random_forest_fare_model.pkl"
preprocessor_path = models_folder / "fare_preprocessor.pkl"

loaded_model = joblib.load(model_path)
loaded_preprocessor = joblib.load(preprocessor_path)

# Create one sample flight
sample_flight = pd.DataFrame({
    "airline": ["Vistara"],
    "source_city": ["Delhi"],
    "departure_time": ["Morning"],
    "stops": ["one"],
    "arrival_time": ["Afternoon"],
    "destination_city": ["Mumbai"],
    "class": ["Economy"],
    "duration": [3.0],
    "days_left": [10],
    "route": ["Delhi_Mumbai"]
})

# Apply the same preprocessing used during training
sample_transformed = loaded_preprocessor.transform(sample_flight)

# Predict fare
predicted_fare = loaded_model.predict(sample_transformed)[0]

print("Flight Details:")
print(sample_flight.to_string(index=False))

print("\nPredicted Fare: ₹", round(predicted_fare, 2))
# %%
