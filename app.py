import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Indian Airline Flight Fare Prediction",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

project_root = Path(__file__).resolve().parent
models_folder = project_root / "models"

model_path = models_folder / "random_forest_fare_model.pkl"
preprocessor_path = models_folder / "fare_preprocessor.pkl"


@st.cache_resource
def load_model_and_preprocessor():
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor


model, preprocessor = load_model_and_preprocessor()


# ============================================================
# CUSTOM UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .result-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #dddddd;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .result-label {
        font-size: 18px;
        margin-bottom: 5px;
    }

    .result-price {
        font-size: 40px;
        font-weight: 700;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">✈️ Indian Airline Flight Fare Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning-based flight fare estimation using historical Indian airline data'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# PROJECT INTRODUCTION
# ============================================================

st.markdown(
    """
    <div class="info-box">

    <b>About this project</b><br><br>

    This system predicts the estimated fare of an Indian domestic flight
    based on airline, route, travel timing, number of stops, class,
    flight duration, and days remaining before departure.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FLIGHT DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">🛫 Flight Details</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    airline = st.selectbox(
        "Airline",
        [
            "Vistara",
            "Air_India",
            "Indigo",
            "GO_FIRST",
            "AirAsia",
            "SpiceJet"
        ]
    )

    source_city = st.selectbox(
        "Source City",
        [
            "Delhi",
            "Mumbai",
            "Bangalore",
            "Kolkata",
            "Hyderabad",
            "Chennai"
        ]
    )


with col2:

    destination_city = st.selectbox(
        "Destination City",
        [
            "Delhi",
            "Mumbai",
            "Bangalore",
            "Kolkata",
            "Hyderabad",
            "Chennai"
        ]
    )

    departure_time = st.selectbox(
        "Departure Time",
        [
            "Morning",
            "Early_Morning",
            "Afternoon",
            "Evening",
            "Night",
            "Late_Night"
        ]
    )


with col3:

    arrival_time = st.selectbox(
        "Arrival Time",
        [
            "Morning",
            "Early_Morning",
            "Afternoon",
            "Evening",
            "Night",
            "Late_Night"
        ]
    )

    stops = st.selectbox(
        "Number of Stops",
        [
            "zero",
            "one",
            "two_or_more"
        ]
    )


# ============================================================
# TRAVEL DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">💺 Travel Details</div>',
    unsafe_allow_html=True
)

col4, col5, col6 = st.columns(3)


with col4:

    flight_class = st.selectbox(
        "Class",
        [
            "Economy",
            "Business"
        ]
    )


with col5:

    duration = st.number_input(
        "Flight Duration (hours)",
        min_value=0.5,
        max_value=50.0,
        value=3.0,
        step=0.1
    )


with col6:

    days_left = st.number_input(
        "Days Left Before Departure",
        min_value=1,
        max_value=49,
        value=10,
        step=1
    )


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

button_col1, button_col2, button_col3 = st.columns([1, 2, 1])

with button_col2:

    predict_button = st.button(
        "✈️ Predict Flight Fare",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # Check source and destination

    if source_city == destination_city:

        st.error(
            "Source City and Destination City cannot be the same."
        )

    else:

        # Create route

        route = source_city + "_" + destination_city

        # Create input dataframe

        sample_flight = pd.DataFrame(
            {
                "airline": [airline],
                "source_city": [source_city],
                "departure_time": [departure_time],
                "stops": [stops],
                "arrival_time": [arrival_time],
                "destination_city": [destination_city],
                "class": [flight_class],
                "duration": [duration],
                "days_left": [days_left],
                "route": [route]
            }
        )

        # Transform input

        sample_transformed = preprocessor.transform(
            sample_flight
        )

        # Predict fare

        predicted_fare = model.predict(
            sample_transformed
        )[0]


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.markdown(
            """
            <div class="result-box">
                <div class="result-label">
                    💰 Estimated Flight Fare
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            label="Predicted Fare",
            value=f"₹ {predicted_fare:,.2f}"
        )


        # ====================================================
        # FLIGHT SUMMARY
        # ====================================================

        st.markdown(
            '<div class="section-title">📋 Flight Summary</div>',
            unsafe_allow_html=True
        )

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:

            st.write(f"**Airline:** {airline}")
            st.write(f"**Route:** {source_city} → {destination_city}")
            st.write(f"**Class:** {flight_class}")

        with summary_col2:

            st.write(f"**Stops:** {stops}")
            st.write(f"**Duration:** {duration} hours")
            st.write(f"**Departure:** {departure_time}")

        with summary_col3:

            st.write(f"**Arrival:** {arrival_time}")
            st.write(f"**Days Left:** {days_left}")
            st.write(f"**Route Feature:** {route}")


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.divider()

        st.markdown(
            """
            ### 🤖 Machine Learning Model

            This prediction is generated using a **Random Forest
            Regression model** trained on historical Indian domestic
            flight data.

            **Model evaluation:**

            - Test R²: **0.985**
            - Test MAE: approximately **₹1,085**
            - Test RMSE: approximately **₹2,808**

            The model was selected after comparing multiple regression
            algorithms.
            """
        )


# ============================================================
# FOOTER / DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "⚠️ The predicted fare is an ML-based estimate from historical data. "
    "It is not a live airline ticket price and may differ from the actual "
    "booking price."
)

st.caption(
    "Indian Airline Flight Fare Prediction System | Machine Learning + Streamlit"
)