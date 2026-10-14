# Indian Airline Flight Fare Prediction System ✈️

A machine learning project that predicts estimated flight fares for domestic Indian airline journeys using historical flight data.

## 📌 Project Overview

The goal of this project is to estimate flight ticket prices based on flight and travel details. A user can enter their journey information through a Streamlit web application and receive an estimated fare.

**Project type:** Machine Learning — Regression

## 🎯 Features

- Exploratory Data Analysis (EDA)
- Data cleaning and feature engineering
- Flight route feature creation
- Categorical feature encoding and numerical feature scaling
- Comparison of multiple regression algorithms
- Random Forest model evaluation
- Actual vs. predicted fare visualization
- Residual error analysis
- Interactive Streamlit application

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Streamlit

## 📊 Dataset

The dataset contains approximately 300,153 flight records and includes features such as:

- Airline
- Source and destination cities
- Departure and arrival times
- Number of stops
- Travel class
- Flight duration
- Days left before departure
- Ticket price

The target variable is `price`.

*Note: The dataset is not included in this repository. Obtain the dataset separately before running the training scripts.*

## ⚙️ Machine Learning Workflow

1. Load and understand the dataset.
2. Clean and explore the data.
3. Create the route feature.
4. Split the data into training and testing sets.
5. Encode categorical features using One-Hot Encoding.
6. Scale numerical features using StandardScaler.
7. Train and compare regression models.
8. Evaluate models using MAE, RMSE, and R².
9. Save the final model and preprocessing object.
10. Use Streamlit to display fare predictions.

## 🤖 Models Compared

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- HistGradientBoosting Regressor

Random Forest Regressor was selected based on its performance among the tested configurations.

### Final Model Performance

| Metric | Test result |
|---|---:|
| MAE | ₹1,084.57 |
| RMSE | ₹2,808.00 |
| R² | 0.9847 |

These are results from the project's test dataset. They do not guarantee the same performance on future flight prices.

## 🚀 Run the Streamlit Application

### 1. Clone the repository

```bash
git clone https://github.com/rahulkumaryenugula0308-dev/Indian-Airline-Fare-Prediction.git
cd Indian-Airline-Fare-Prediction
```

### 2. Install the dependencies

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

### 3. Add the saved model files

Place the trained model and fitted preprocessor in the `models` folder:

- `random_forest_fare_model.pkl`
- `fare_preprocessor.pkl`

These files are excluded from this repository, so they must be supplied separately.

### 4. Start the application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 📈 Visualizations

The `reports` folder contains visualizations created during exploratory data analysis, including flight distributions, price comparisons, route analysis, and correlation analysis.

## 🔮 Future Improvements

- Use more recent flight fare data.
- Improve feature engineering.
- Perform cross-validation and additional hyperparameter tuning.
- Investigate large prediction errors.
- Compare additional regression algorithms.
- Explore deployment through a cloud hosting platform.

## ⚠️ Disclaimer

This application estimates fares from historical data. It does not fetch live airline prices and should not be used as a guaranteed booking price.

## 👨‍💻 Project Summary

This project demonstrates a complete machine learning workflow, from data exploration and preprocessing to model evaluation and deployment through an interactive Streamlit application.
