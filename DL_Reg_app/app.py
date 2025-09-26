import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px

# Page setup
st.set_page_config(
    page_title="Solar Power Generation Predictor",
    page_icon="☀️",
    layout="wide"
)

# Main title
st.title("☀️ Solar Power Generation Predictor")
st.markdown("---")

# Function to load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = keras.models.load_model("my_model.keras")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load model and scaler
model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error("Model files not found. Make sure my_model.keras and scaler.pkl are in the same folder")
    st.stop()

# Create input columns
st.sidebar.header("Input Settings")
st.sidebar.info("Enter the following values to get power generation prediction")

# Create input columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Input Data")
    
    # Basic variables
    solar_irradiance = st.number_input(
        "Solar Irradiance (W/m²)", 
        min_value=0.0, 
        max_value=1500.0, 
        value=800.0,
        step=10.0
    )
    
    temperature = st.number_input(
        "Temperature (°C)", 
        min_value=-20.0, 
        max_value=60.0, 
        value=25.0,
        step=0.1
    )
    
    humidity = st.number_input(
        "Humidity (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=50.0,
        step=1.0
    )
    
    wind_speed = st.number_input(
        "Wind Speed (m/s)", 
        min_value=0.0, 
        max_value=30.0, 
        value=5.0,
        step=0.1
    )
    
    panel_area = st.number_input(
        "Panel Area (m²)", 
        min_value=1.0, 
        max_value=1000.0, 
        value=50.0,
        step=1.0
    )
    
    # Additional variables
    st.subheader("Additional Variables")
    
    pressure = st.number_input(
        "Atmospheric Pressure (hPa)", 
        min_value=900.0, 
        max_value=1100.0, 
        value=1013.0,
        step=1.0
    )
    
    cloud_cover = st.number_input(
        "Cloud Cover (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=20.0,
        step=1.0
    )
    
    visibility = st.number_input(
        "Visibility (km)", 
        min_value=0.1, 
        max_value=50.0, 
        value=10.0,
        step=0.1
    )
    
    dew_point = st.number_input(
        "Dew Point (°C)", 
        min_value=-30.0, 
        max_value=30.0, 
        value=15.0,
        step=0.1
    )
    
    uv_index = st.number_input(
        "UV Index", 
        min_value=0.0, 
        max_value=15.0, 
        value=6.0,
        step=0.1
    )
    
    wind_direction = st.number_input(
        "Wind Direction (degrees)", 
        min_value=0.0, 
        max_value=360.0, 
        value=180.0,
        step=1.0
    )
    
    panel_temperature = st.number_input(
        "Panel Temperature (°C)", 
        min_value=-20.0, 
        max_value=80.0, 
        value=35.0,
        step=0.1
    )
    
    panel_efficiency = st.number_input(
        "Panel Efficiency (%)", 
        min_value=10.0, 
        max_value=25.0, 
        value=18.0,
        step=0.1
    )
    
    inverter_efficiency = st.number_input(
        "Inverter Efficiency (%)", 
        min_value=80.0, 
        max_value=99.0, 
        value=95.0,
        step=0.1
    )
    
    dust_factor = st.number_input(
        "Dust Factor (%)", 
        min_value=0.0, 
        max_value=30.0, 
        value=5.0,
        step=0.1
    )
    
    # Time variables
    hour_of_day = st.number_input(
        "Hour of Day (0-23)", 
        min_value=0, 
        max_value=23, 
        value=12,
        step=1
    )
    
    day_of_year = st.number_input(
        "Day of Year (1-365)", 
        min_value=1, 
        max_value=365, 
        value=180,
        step=1
    )
    
    sun_elevation = st.number_input(
        "Sun Elevation Angle (degrees)", 
        min_value=0.0, 
        max_value=90.0, 
        value=45.0,
        step=1.0
    )
    
    sun_azimuth = st.number_input(
        "Sun Azimuth Angle (degrees)", 
        min_value=0.0, 
        max_value=360.0, 
        value=180.0,
        step=1.0
    )
    
    ambient_light = st.number_input(
        "Ambient Light (lux)", 
        min_value=0.0, 
        max_value=120000.0, 
        value=50000.0,
        step=1000.0
    )

with col2:
    st.subheader("🔮 Results")
    
    if st.button("Predict Power Generation", type="primary"):
        try:
            # Create input array with all 20 features
            input_data = np.array([[
                solar_irradiance,
                temperature,
                humidity,
                wind_speed,
                panel_area,
                pressure,
                cloud_cover,
                visibility,
                dew_point,
                uv_index,
                wind_direction,
                panel_temperature,
                panel_efficiency,
                inverter_efficiency,
                dust_factor,
                hour_of_day,
                day_of_year,
                sun_elevation,
                sun_azimuth,
                ambient_light
            ]])
            
            # Apply scaling
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            predicted_power = prediction[0][0]
            
            # Display result
            st.success(f"Predicted Power: **{predicted_power:.2f} kW**")
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = predicted_power,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Power Generation (kW)"},
                gauge = {
                    'axis': {'range': [None, max(500, predicted_power * 1.5)]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, predicted_power * 0.5], 'color': "lightgray"},
                        {'range': [predicted_power * 0.5, predicted_power * 0.8], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': predicted_power * 0.9
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")

# CSV file upload section
st.markdown("---")
st.subheader("📁 CSV File Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV file for batch prediction", 
    type=['csv']
)

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        st.write("Data sample:")
        st.dataframe(df_input.head())
        
        if st.button("Run Prediction on File"):
            # Apply scaling
            X_scaled = scaler.transform(df_input)
            
            # Make predictions
            predictions = model.predict(X_scaled)
            
            # Add predictions to data
            df_results = df_input.copy()
            df_results['predicted_power_kw'] = predictions
            
            st.success("Prediction completed successfully!")
            st.dataframe(df_results)
            
            # Create chart for results
            fig = px.line(
                df_results, 
                y='predicted_power_kw',
                title="Predictions Over Time",
                labels={'predicted_power_kw': 'Predicted Power (kW)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Download option
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv'
            )
            
    except Exception as e:
        st.error(f"File processing error: {e}")

# App info
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Solar Power Generation Predictor v1.0</p>
    <p>Built with Streamlit and TensorFlow</p>
</div>
""", unsafe_allow_html=True)
