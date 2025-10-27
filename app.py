import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="NASA Turbofan RUL Predictor",
    page_icon="✈️",
    layout="centered"
)

# Loading Model
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("gradient_boost_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# Page Style
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        padding: 2rem;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #0056b3;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Page Title
st.title("✈️ NASA Turbofan Remaining Useful Life (RUL) Prediction")
st.write("Predict the remaining useful life of a turbofan engine based on key sensor readings.")

st.markdown("---")

# Input 
st.header("🧩 Input Engine Sensor Readings")

col1, col2 = st.columns(2)

with col1:
    time_in_cycles = st.number_input("Time in Cycles")
    sensor_2 = st.number_input("Sensor 2")
    sensor_3 = st.number_input("Sensor 3")
    sensor_4 = st.number_input("Sensor 4")
    sensor_7 = st.number_input("Sensor 7")
    sensor_8 = st.number_input("Sensor 8")

with col2:
    sensor_9 = st.number_input("Sensor 9")
    sensor_15 = st.number_input("Sensor 15")
    sensor_17 = st.number_input("Sensor 17")
    sensor_20 = st.number_input("Sensor 20")
    sensor_21 = st.number_input("Sensor 21")

# Prepare input DataFrame
input_data = pd.DataFrame({
    'time_in_cycles': [time_in_cycles],
    'sensor_2': [sensor_2],
    'sensor_3': [sensor_3],
    'sensor_4': [sensor_4],
    'sensor_7': [sensor_7],
    'sensor_8': [sensor_8],
    'sensor_9': [sensor_9],
    'sensor_15': [sensor_15],
    'sensor_17': [sensor_17],
    'sensor_20': [sensor_20],
    'sensor_21': [sensor_21]
})

# Prediction
if st.button("🔍 Predict RUL"):
    try:
        # Scale using saved scaler
        input_scaled = scaler.transform(input_data)

        # Drop engine_id before prediction
        scaled_df = pd.DataFrame(input_scaled, columns=input_data.columns)

        # Predict
        prediction = int(model.predict(scaled_df)[0])

        st.success(f"🛠 Estimated Remaining Useful Life: **{prediction:.2f} cycles**")

        if prediction < 25:
            st.warning("⚠️ Engine nearing failure — consider maintenance soon.")
        elif prediction < 100:
            st.info("🧭 Moderate wear detected — schedule inspection.")
        else:
            st.success("✅ Engine operating well within safe limits.")

    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")


# Footer 
st.markdown("---")
st.subheader("📘 About this App")
st.write("""
This interactive web app uses a **Gradient Boosting Regression model** trained on NASA’s **CMAPSS Turbofan Engine dataset (FD001)**.  
It predicts the **Remaining Useful Life (RUL)** of an engine based on real sensor readings.

**Features:**
- Automatically scales user inputs before prediction  
- Real-time RUL estimation and visual interpretation  
- Extendable to other datasets (FD002, FD003, etc.)  
""")
