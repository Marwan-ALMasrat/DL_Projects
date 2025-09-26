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
        return None, None

# Load model and scaler
model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error("⚠️ Model files not found. Please make sure 'my_model.keras' and 'scaler.pkl' are in the same directory as this script.")
    st.info("📁 Expected file structure:\n- app.py\n- my_model.keras\n- scaler.pkl")
    st.stop()

# Create input columns
st.sidebar.header("Input Settings")
st.sidebar.info("Enter the following values to get power generation prediction")

# Create input columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Weather Data")
    
    # Temperature and humidity
    temperature_2_m_above_gnd = st.number_input(
        "Temperature 2m Above Ground (°C)", 
        min_value=-40.0, 
        max_value=60.0, 
        value=25.0,
        step=0.1
    )
    
    relative_humidity_2_m_above_gnd = st.number_input(
        "Relative Humidity 2m Above Ground (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=50.0,
        step=1.0
    )
    
    # Pressure and precipitation
    mean_sea_level_pressure_MSL = st.number_input(
        "Mean Sea Level Pressure (hPa)", 
        min_value=900.0, 
        max_value=1100.0, 
        value=1013.25,
        step=0.1
    )
    
    total_precipitation_sfc = st.number_input(
        "Total Precipitation Surface (mm)", 
        min_value=0.0, 
        max_value=200.0, 
        value=0.0,
        step=0.1
    )
    
    snowfall_amount_sfc = st.number_input(
        "Snowfall Amount Surface (mm)", 
        min_value=0.0, 
        max_value=500.0, 
        value=0.0,
        step=0.1
    )
    
    # Cloud coverage
    st.subheader("☁️ Cloud Coverage")
    
    total_cloud_cover_sfc = st.number_input(
        "Total Cloud Cover Surface (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=20.0,
        step=1.0
    )
    
    high_cloud_cover_high_cld_lay = st.number_input(
        "High Cloud Cover (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=5.0,
        step=1.0
    )
    
    medium_cloud_cover_mid_cld_lay = st.number_input(
        "Medium Cloud Cover (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=10.0,
        step=1.0
    )
    
    low_cloud_cover_low_cld_lay = st.number_input(
        "Low Cloud Cover (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=15.0,
        step=1.0
    )
    
    # Solar radiation
    st.subheader("☀️ Solar Radiation")
    
    shortwave_radiation_backwards_sfc = st.number_input(
        "Shortwave Radiation Backwards Surface (W/m²)", 
        min_value=0.0, 
        max_value=1500.0, 
        value=800.0,
        step=10.0
    )

with col2:
    st.subheader("🌬️ Wind Data")
    
    # Wind at 10m
    wind_speed_10_m_above_gnd = st.number_input(
        "Wind Speed 10m Above Ground (m/s)", 
        min_value=0.0, 
        max_value=50.0, 
        value=5.0,
        step=0.1
    )
    
    wind_direction_10_m_above_gnd = st.number_input(
        "Wind Direction 10m Above Ground (degrees)", 
        min_value=0.0, 
        max_value=360.0, 
        value=180.0,
        step=1.0
    )
    
    # Wind at 80m
    wind_speed_80_m_above_gnd = st.number_input(
        "Wind Speed 80m Above Ground (m/s)", 
        min_value=0.0, 
        max_value=70.0, 
        value=8.0,
        step=0.1
    )
    
    wind_direction_80_m_above_gnd = st.number_input(
        "Wind Direction 80m Above Ground (degrees)", 
        min_value=0.0, 
        max_value=360.0, 
        value=180.0,
        step=1.0
    )
    
    # Wind at 900mb
    wind_speed_900_mb = st.number_input(
        "Wind Speed 900mb (m/s)", 
        min_value=0.0, 
        max_value=100.0, 
        value=10.0,
        step=0.1
    )
    
    wind_direction_900_mb = st.number_input(
        "Wind Direction 900mb (degrees)", 
        min_value=0.0, 
        max_value=360.0, 
        value=180.0,
        step=1.0
    )
    
    wind_gust_10_m_above_gnd = st.number_input(
        "Wind Gust 10m Above Ground (m/s)", 
        min_value=0.0, 
        max_value=80.0, 
        value=7.0,
        step=0.1
    )
    
    st.subheader("🌞 Solar Angles")
    
    # Solar angles
    angle_of_incidence = st.number_input(
        "Angle of Incidence (degrees)", 
        min_value=0.0, 
        max_value=90.0, 
        value=30.0,
        step=0.1
    )
    
    zenith = st.number_input(
        "Zenith Angle (degrees)", 
        min_value=0.0, 
        max_value=180.0, 
        value=45.0,
        step=0.1
    )
    
    azimuth = st.number_input(
        "Azimuth Angle (degrees)", 
        min_value=0.0, 
        max_value=360.0, 
        value=180.0,
        step=0.1
    )
    
    st.subheader("🔮 Results")
    
    if st.button("Predict Power Generation", type="primary"):
        try:
            # Create input array with exact feature names from your model
            input_data = np.array([[
                temperature_2_m_above_gnd,
                relative_humidity_2_m_above_gnd,
                mean_sea_level_pressure_MSL,
                total_precipitation_sfc,
                snowfall_amount_sfc,
                total_cloud_cover_sfc,
                high_cloud_cover_high_cld_lay,
                medium_cloud_cover_mid_cld_lay,
                low_cloud_cover_low_cld_lay,
                shortwave_radiation_backwards_sfc,
                wind_speed_10_m_above_gnd,
                wind_direction_10_m_above_gnd,
                wind_speed_80_m_above_gnd,
                wind_direction_80_m_above_gnd,
                wind_speed_900_mb,
                wind_direction_900_mb,
                wind_gust_10_m_above_gnd,
                angle_of_incidence,
                zenith,
                azimuth
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
            st.error("Please check if the input features match your trained model")

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
        
        st.write("Expected column order:")
        expected_columns = [
            "temperature_2_m_above_gnd", "relative_humidity_2_m_above_gnd", 
            "mean_sea_level_pressure_MSL", "total_precipitation_sfc", 
            "snowfall_amount_sfc", "total_cloud_cover_sfc", 
            "high_cloud_cover_high_cld_lay", "medium_cloud_cover_mid_cld_lay", 
            "low_cloud_cover_low_cld_lay", "shortwave_radiation_backwards_sfc", 
            "wind_speed_10_m_above_gnd", "wind_direction_10_m_above_gnd", 
            "wind_speed_80_m_above_gnd", "wind_direction_80_m_above_gnd", 
            "wind_speed_900_mb", "wind_direction_900_mb", 
            "wind_gust_10_m_above_gnd", "angle_of_incidence", "zenith", "azimuth"
        ]
        st.write(expected_columns)
        
        if st.button("Run Prediction on File"):
            # Check if we have the right number of columns
            if df_input.shape[1] != 20:
                st.error(f"Expected 20 columns but got {df_input.shape[1]}. Please check your CSV file.")
            else:
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
