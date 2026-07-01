import streamlit as st
import gridstatus
import pandas as pd
import requests
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="CAISO Analytics Hub", layout="wide")

st.title("California Energy Dashboard")
st.write("Tracking real-time grid generation, market economics, weather correlations, and machine learning load forecasting.")



@st.cache_data
def fetch_caiso_data():
    caiso = gridstatus.CAISO()
    df = caiso.get_fuel_mix(date="today")
    df = df.select_dtypes(include='number').join(df[['Time']])
    df.set_index("Time", inplace=True)
    return df
@st.cache_data
def fetch_weather_data():
    #SMUD Territory
    url = "https://api.open-meteo.com/v1/forecast?latitude=38.5816&longitude=-121.4944&hourly=temperature_2m&timezone=America%2FLos_Angeles"
    response = requests.get(url).json()
    weather_df = pd.DataFrame(response["hourly"])
    weather_df["time"] = pd.to_datetime(weather_df["time"])
    weather_df.set_index("time", inplace=True)
    return weather_df
def fetch_load_data():
    caiso = gridstatus.CAISO()
    load_df = caiso.get_load(date="today")
    load_df.set_index("Time", inplace=True)
    return load_df

with st.spinner("Initializing ETL Pipelines & ML Models..."):
    mix_df = fetch_caiso_data()
    weather_df = fetch_weather_data()
    load_df = fetch_load_data()
    
tab1, tab2, tab3 = st.tabs(["Live Grid & Carbon", "Weather Correlation", "AI Load Forecast"])


###TAB 1 Live Grid and Carbon Intensity

with tab1:
    st.subheader("Current Grid Status & Carbon Tracking")
    numeric_df = mix_df.select_dtypes(include='number')
    latest_data = numeric_df.iloc[-1]
    total_gen = latest_data.sum()
    clean_gen = (
        latest_data.get("Solar", 0) + latest_data.get("Wind", 0) + 
        latest_data.get("Large Hydro", 0) + latest_data.get("Small hydro", 0) + 
        latest_data.get("Geothermal", 0) + latest_data.get("Nuclear", 0)
    )
    clean_percent = (clean_gen / total_gen) * 100 if total_gen > 0 else 0
    # Carbon Math: Approx 0.43 metric tons of CO2 per MWh of Natural Gas
    gas_gen = latest_data.get("Natural Gas", 0)
    co2_emissions = gas_gen * 0.43 
    carbon_intensity = (co2_emissions / total_gen) * 1000 if total_gen > 0 else 0 # kg CO2 per MWh

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Demand (MW)", f"{total_gen:,.0f}")
    col2.metric("Clean Energy", f"{clean_percent:.1f}%")
    col3.metric("Battery Output (MW)", f"{latest_data.get('Batteries', 0):,.0f}")
    col4.metric("Carbon Intensity (kg/MWh)", f"{carbon_intensity:.1f}")

    st.divider()
    
    # Net Load Calculation
    mix_df["Total Generation"] = numeric_df.sum(axis=1)
    mix_df["Net Load"] = mix_df["Total Generation"] - (mix_df.get("Solar", 0) + mix_df.get("Wind", 0))
    
    columns_to_plot = ["Solar", "Net Load", "Natural Gas", "Batteries", "Large Hydro", "Nuclear"]
    st.line_chart(mix_df[columns_to_plot])

#TAB2 Weather Correlation
with tab2:
    st.subheader("Demand vs. Temperature (Sacramento)")
    st.write("Merging CAISO grid demand with Open-Meteo weather APIs to track temperature correlation.")
    
    hourly_load = load_df["Load"].resample("h").mean()
    hourly_temp = weather_df["temperature_2m"]
    
    weather_col1, weather_col2 = st.columns(2)
    with weather_col1:
        st.markdown("**Grid Demand (MW)**")
        st.line_chart(hourly_load)
    with weather_col2:
        st.markdown("**Temperature (°C)**")
        st.line_chart(hourly_temp, color="#ffaa00")
        
#TAB 3 Maching learning Forecast
with tab3:
    st.subheader("Predictive Demand Modeling (Next 24 Hours)")
    st.write("Using a Random Forest Regressor trained on today's grid patterns to forecast tomorrow's load baseline.")
    
    # 1. Prepare Data for ML 
    df_ml = load_df.copy()
    df_ml["Hour"] = df_ml.index.hour
    df_ml["Minute"] = df_ml.index.minute
    
    # 2. Train the Model
    X = df_ml[["Hour", "Minute"]]
    y = df_ml["Load"]
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # 3. Predict tomorrow
    future_times = pd.date_range(start=df_ml.index[-1], periods=288, freq="5min")
    future_X = pd.DataFrame({"Hour": future_times.hour, "Minute": future_times.minute})
    predictions = model.predict(future_X)
    
    # 4. Format the future data
    forecast_df = pd.DataFrame({"Predicted Load": predictions}, index=future_times)
    
    # Rename today's load column so it looks clean on the chart legend
    historical_df = df_ml[["Load"]].rename(columns={"Load": "Actual Load"})
    
    # Merge the past and the future into one master dataframe
    combined_df = pd.concat([historical_df, forecast_df])
    
    # Plot both lines together! Streamlit will automatically assign them different colors
    st.line_chart(combined_df)

##with st.spinner("Pulling live data from CAISO..."):
##    mix_df = fetch_caiso_data()
st.subheader("Live Generation Mix (Megawatts)")
st.dataframe(mix_df.tail())
##
##st.subheader("Today's Generation Trends")
##st.write("Notice how Solar drops off in the evening, causing Natural Gas and Batteries to spike.")

##st.header("Current Grid Status")
##numeric_df = mix_df.select_dtypes(include='number')
##latest_data = numeric_df.iloc[-1]

total_gen = latest_data.sum()
clean_gen = (
    latest_data.get("Solar", 0) + 
    latest_data.get("Wind", 0) + 
    latest_data.get("Large Hydro", 0) +
    latest_data.get("Small Hydro", 0) + 
    latest_data.get("Geothermal", 0) + 
    latest_data.get("Nuclear", 0)
)

#Percent of Clean Energy
clean_total_percent = (clean_gen/total_gen) * 100



#st.divider()

columns_to_plot = [
    "Solar", 
    "Natural Gas", 
    "Batteries", 
    "Large Hydro", 
    "Nuclear", 
    "Geothermal", 
    "Small Hydro"
]

##st.line_chart(mix_df[columns_to_plot])



# --WHOLESALE PRICING & ARBITRAGE SECTION ---
st.divider()
st.subheader("Battery Arbitrage & Market Pricing")
st.write("Comparing real-time market costs (LMP) at NP15 to grid battery behavior. Notice how batteries discharge (generate power) exactly when prices spike.")

@st.cache_data
def fetch_pricing_data():
    caiso = gridstatus.CAISO()
    pricing_df = caiso.get_lmp(date="today", market="REAL_TIME_5_MIN", locations=["TH_NP15_GEN-APND"])
    pricing_df = pricing_df.select_dtypes(include='number').join(pricing_df[['Time']])
    pricing_df.set_index("Time", inplace=True)
    return pricing_df

with st.spinner("Fetching market pricing..."):
    price_df = fetch_pricing_data()

price_col1, price_col2 = st.columns(2)

with price_col1:
    st.markdown("**1. Market Price (LMP)**")
    st.line_chart(price_df["LMP"], color="#ff2b2b") # Red line for prices

with price_col2:
    st.markdown("**2. Battery Output**")
    st.line_chart(mix_df["Batteries"], color="#00ff00") # Green line for clean batteries