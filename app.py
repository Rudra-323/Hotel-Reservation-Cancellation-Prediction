
# Hotel Reservation Cancellation Predictor — Streamlit App


import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

# ----------------------------------------------------------------------------
# 1. PAGE CONFIG & UI SETUP
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hotel Reservation Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
)

st.title("🏨 Hotel Reservation Cancellation Predictor")
st.markdown(
    """
    #### Estimate the likelihood that an incoming reservation will be cancelled
    This tool uses a trained **Artificial Neural Network (ANN)** to score a hotel booking's
    cancellation risk from its details (lead time, price, guest composition, market segment,
    etc.). Fill in the booking details in the **sidebar**, then click
    **"Predict Cancellation Risk"** to see the model's assessment and a suggested next action.
    """
)
st.divider()

# ----------------------------------------------------------------------------
# ARTIFACT PATHS
# ----------------------------------------------------------------------------
ARTIFACT_DIR = "artifacts"
MODEL_PATH_KERAS = os.path.join(ARTIFACT_DIR, "hotel_cancellation_ann.keras")
MODEL_PATH_H5 = os.path.join(ARTIFACT_DIR, "hotel_cancellation_ann.h5")
PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.joblib")


# ----------------------------------------------------------------------------
# 3. PREDICTION PIPELINE — CACHED ARTIFACT LOADING WITH ERROR HANDLING
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model and preprocessor...")
def load_artifacts():
    """
    Load the trained Keras ANN and the fitted preprocessing pipeline.
    Cached with st.cache_resource so this only runs once per session, not on
    every Streamlit rerun (e.g. every widget interaction).

    Returns:
        (model, preprocessor) tuple, or (None, None) if loading fails.
    """
    # --- Load model: prefer native .keras format, fall back to legacy .h5 ---
    model = None
    for path in (MODEL_PATH_KERAS, MODEL_PATH_H5):
        if os.path.exists(path):
            try:
                model = keras.models.load_model(path)
                break
            except Exception as e:
                st.error(f"Found model file at `{path}` but failed to load it: {e}")
                return None, None

    if model is None:
        st.error(
            "❌ Could not find a trained model. Expected "
            f"`{MODEL_PATH_KERAS}` or `{MODEL_PATH_H5}`. "
            "Please train the model and place the exported file in the `artifacts/` folder."
        )
        return None, None

    # --- Load fitted preprocessor (ColumnTransformer: OneHotEncoder + StandardScaler) ---
    if not os.path.exists(PREPROCESSOR_PATH):
        st.error(
            f"❌ Could not find the preprocessor at `{PREPROCESSOR_PATH}`. "
            "Please export the fitted ColumnTransformer with joblib and place it in "
            "the `artifacts/` folder."
        )
        return None, None

    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    except Exception as e:
        st.error(f"Found preprocessor file but failed to load it: {e}")
        return None, None

    return model, preprocessor


model, preprocessor = load_artifacts()

# Stop the app here if artifacts failed to load — no point rendering a form
# that can't produce a prediction.
if model is None or preprocessor is None:
    st.warning(
        "The app cannot make predictions until both the model and preprocessor "
        "artifacts are available. Add them to the `artifacts/` folder and refresh the page."
    )
    st.stop()


# ----------------------------------------------------------------------------
# 2. SIDEBAR INPUT FORM
# ----------------------------------------------------------------------------
st.sidebar.header("📝 Reservation Details")
st.sidebar.caption("Fill in the booking details below, then scroll down and click predict.")

with st.sidebar.form("booking_form"):

    # --- Booking Details ---
    with st.expander("🛏️ Booking Details", expanded=True):
        no_of_adults = st.slider("Number of Adults", min_value=0, max_value=10, value=2)
        no_of_children = st.slider("Number of Children", min_value=0, max_value=10, value=0)
        no_of_weekend_nights = st.slider(
            "Number of Weekend Nights", min_value=0, max_value=10, value=1
        )
        no_of_week_nights = st.slider(
            "Number of Week Nights", min_value=0, max_value=20, value=2
        )

    # --- Plan & Room Info ---
    with st.expander("🍽️ Plan & Room Info", expanded=True):
        type_of_meal_plan = st.selectbox(
            "Meal Plan",
            ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"],
        )
        room_type_reserved = st.selectbox(
            "Room Type Reserved",
            [f"Room_Type {i}" for i in range(1, 8)],
        )
        required_car_parking_space = st.selectbox(
            "Requires Car Parking Space?", options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

    # --- Booking Metrics ---
    with st.expander("📊 Booking Metrics", expanded=True):
        lead_time = st.slider(
            "Lead Time (days between booking and arrival)",
            min_value=0, max_value=500, value=45,
        )

        col_y, col_m, col_d = st.columns(3)
        with col_y:
            arrival_year = st.number_input(
                "Arrival Year", min_value=2017, max_value=2030, value=2018, step=1
            )
        with col_m:
            arrival_month = st.number_input(
                "Arrival Month", min_value=1, max_value=12, value=7, step=1
            )
        with col_d:
            arrival_date = st.number_input(
                "Arrival Date", min_value=1, max_value=31, value=15, step=1
            )

        market_segment_type = st.selectbox(
            "Market Segment Type",
            ["Online", "Offline", "Corporate", "Aviation", "Complementary"],
        )
        repeated_guest = st.selectbox(
            "Repeated Guest?", options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        no_of_previous_cancellations = st.number_input(
            "Number of Previous Cancellations", min_value=0, max_value=50, value=0, step=1
        )
        no_of_previous_bookings_not_canceled = st.number_input(
            "Number of Previous Bookings Not Canceled",
            min_value=0, max_value=100, value=0, step=1,
        )
        # Monetary value -> plain numeric input rather than a slider
        avg_price_per_room = st.number_input(
            "Average Price per Room ($)", min_value=0.0, max_value=1000.0,
            value=105.50, step=0.5, format="%.2f",
        )
        no_of_special_requests = st.slider(
            "Number of Special Requests", min_value=0, max_value=10, value=1
        )

    st.markdown("---")
    submitted = st.form_submit_button("🔮 Predict Cancellation Risk", use_container_width=True)


# ----------------------------------------------------------------------------
# 3. (continued) BUILD INPUT DATAFRAME + RUN PREDICTION
# ----------------------------------------------------------------------------
def build_input_dataframe() -> pd.DataFrame:
    """
    Assemble all sidebar form inputs into a single-row Pandas DataFrame with
    column names and dtypes matching the original training dataset, so it can
    be passed straight into the fitted preprocessor.
    """
    return pd.DataFrame([{
        "no_of_adults": no_of_adults,
        "no_of_children": no_of_children,
        "no_of_weekend_nights": no_of_weekend_nights,
        "no_of_week_nights": no_of_week_nights,
        "type_of_meal_plan": type_of_meal_plan,
        "required_car_parking_space": required_car_parking_space,
        "room_type_reserved": room_type_reserved,
        "lead_time": lead_time,
        "arrival_year": arrival_year,
        "arrival_month": arrival_month,
        "arrival_date": arrival_date,
        "market_segment_type": market_segment_type,
        "repeated_guest": repeated_guest,
        "no_of_previous_cancellations": no_of_previous_cancellations,
        "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
        "avg_price_per_room": avg_price_per_room,
        "no_of_special_requests": no_of_special_requests,
    }])


def predict_cancellation_probability(input_df: pd.DataFrame) -> float:
    """
    Apply the fitted preprocessor to raw input, then run the ANN to get a
    cancellation probability in [0, 1].
    """
    # Transform raw features using the SAME fitted encoder/scaler used in training
    processed = preprocessor.transform(input_df)

    # ColumnTransformer may return a sparse matrix depending on sklearn version
    if hasattr(processed, "toarray"):
        processed = processed.toarray()

    # model.predict returns a (1, 1) array of sigmoid probabilities
    probability = model.predict(processed, verbose=0).ravel()[0]
    return float(probability)


# ----------------------------------------------------------------------------
# 4. OUTPUT DASHBOARD
# ----------------------------------------------------------------------------
if submitted:
    input_df = build_input_dataframe()

    with st.spinner("Scoring reservation..."):
        try:
            probability = predict_cancellation_probability(input_df)
        except Exception as e:
            st.error(
                "⚠️ Something went wrong while generating the prediction. This usually means "
                "the input columns don't match what the preprocessor was fitted on.\n\n"
                f"Error details: {e}"
            )
            st.stop()

    st.subheader("📈 Prediction Result")

    result_col, detail_col = st.columns([1, 2])

    with result_col:
        st.metric(label="Cancellation Probability", value=f"{probability:.1%}")
        # Simple visual gauge using a progress bar (0-100 scale)
        st.progress(min(max(probability, 0.0), 1.0))

    with detail_col:
        if probability > 0.5:
            st.warning("🚨 **High Risk of Cancellation**", icon="⚠️")
            st.markdown(
                f"""
                This reservation has an estimated **{probability:.1%} probability of
                cancellation**, which is above the 50% risk threshold.

                **Recommended actions:**
                - Send a confirmation check-in email to the guest closer to the arrival date.
                - Consider requiring a non-refundable deposit or prepayment.
                - Flag this booking for a follow-up call from the reservations team.
                - Factor this booking into overbooking / waitlist planning for that date.
                """
            )
        else:
            st.success("✅ **Low Risk / Likely Confirmed Booking**", icon="✅")
            st.markdown(
                f"""
                This reservation has an estimated **{probability:.1%} probability of
                cancellation**, which is below the 50% risk threshold.

                **Recommended actions:**
                - Proceed with standard guest welcoming and pre-arrival procedures.
                - No additional deposit or confirmation call is likely necessary.
                - Include this booking normally in occupancy and staffing forecasts.
                """
            )

    with st.expander("🔍 View the exact input sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

else:
    st.info("👈 Fill in the reservation details in the sidebar and click **Predict Cancellation Risk** to see results.")

st.divider()
st.caption(
    "Model: TensorFlow/Keras ANN (Dense 64 → 32 → 16 → 1, ReLU/Sigmoid, Dropout) · "
    "Preprocessing: One-Hot Encoding + StandardScaler, fitted on training data only."
)
