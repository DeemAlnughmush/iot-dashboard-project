import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="IoT Sensor Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv("clean_iot_data.csv")
pred_df = pd.read_csv("predictions.csv")
metrics_df = pd.read_csv("metrics.csv")

df["ts"] = pd.to_datetime(df["ts"])

# ---------------------------
# Title
# ---------------------------
st.title("IoT Sensor Data Prediction Dashboard")
st.markdown(
    "Interactive dashboard for analyzing IoT environmental sensor trends "
    "and predicting temperature values using machine learning."
)

# ---------------------------
# Sidebar filters
# ---------------------------
st.sidebar.header("Dashboard Filters")
st.sidebar.markdown(
    "Use the filters below to explore the sensor data by date, hour, "
    "sensor type, and environmental conditions."
)

min_date = df["ts"].min().date()
max_date = df["ts"].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date],
    help="Choose the start and end dates to filter the displayed sensor readings."
)

sensor_labels = {
    "temp": "Temperature",
    "humidity": "Humidity",
    "co": "CO Gas",
    "lpg": "LPG Gas",
    "smoke": "Smoke Level"
}

sensor_descriptions = {
    "temp": "Measures the air temperature.",
    "humidity": "Measures the amount of moisture in the air.",
    "co": "Measures the level of carbon monoxide gas.",
    "lpg": "Measures the level of liquefied petroleum gas.",
    "smoke": "Measures the concentration of smoke in the environment."
}

sensor_option_label = st.sidebar.selectbox(
    "Choose Sensor Type",
    list(sensor_labels.values()),
    help="""
Select which sensor to display in the dynamic trend chart.

Temperature: measures air temperature.
Humidity: measures air moisture.
CO Gas: measures carbon monoxide level.
LPG Gas: measures liquefied petroleum gas level.
Smoke Level: measures smoke concentration.
"""
)

sensor_option = [k for k, v in sensor_labels.items() if v == sensor_option_label][0]

st.sidebar.caption(f"Sensor info: {sensor_descriptions[sensor_option]}")

hour_range = st.sidebar.slider(
    "Select Hour Range",
    0, 23, (0, 23),
    help="Filter data by hour of the day. For example, selecting 8 to 18 shows daytime readings only."
)

light_option = st.sidebar.selectbox(
    "Light Condition",
    ["All", "No Light", "Light Detected"],
    help="Shows whether the light sensor detected light. 'No Light' means the sensor did not detect light, while 'Light Detected' means it did."
)

motion_option = st.sidebar.selectbox(
    "Motion Status",
    ["All", "No Motion", "Motion Detected"],
    help="Shows whether the motion sensor detected movement. 'No Motion' means no movement was detected, while 'Motion Detected' means movement was detected."
)

# ---------------------------
# Team members under filters
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Team Members")
st.sidebar.markdown("""
- Deem Ali Alnughmush – 421201996  
- Fay Alsalhi – 422205635  
- Rahaf Saud Alharbi – 431201565
""")

# ---------------------------
# Filtering
# ---------------------------
filtered = df.copy()

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[
        (filtered["ts"].dt.date >= start_date) &
        (filtered["ts"].dt.date <= end_date)
    ]

filtered = filtered[
    (filtered["hour"] >= hour_range[0]) &
    (filtered["hour"] <= hour_range[1])
]

if light_option == "No Light":
    filtered = filtered[filtered["light"] == 0]
elif light_option == "Light Detected":
    filtered = filtered[filtered["light"] == 1]

if motion_option == "No Motion":
    filtered = filtered[filtered["motion"] == 0]
elif motion_option == "Motion Detected":
    filtered = filtered[filtered["motion"] == 1]

# ---------------------------
# KPI Cards
# ---------------------------
best_model_row = metrics_df.loc[metrics_df["R2"].idxmax()]
best_model = best_model_row["Model"]
best_r2 = best_model_row["R2"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Average Temperature", round(filtered["temp"].mean(), 3))
c2.metric("Average Humidity", round(filtered["humidity"].mean(), 3))
c3.metric("Average Smoke Level", round(filtered["smoke"].mean(), 3))
c4.metric("Best Model", best_model)
c5.metric("Best R²", round(best_r2, 3))

# ---------------------------
# Main charts
# ---------------------------
left, right = st.columns(2)

with left:
    fig_temp = px.line(
        filtered,
        x="ts",
        y="temp",
        title="Temperature Over Time"
    )
    st.plotly_chart(fig_temp, use_container_width=True, key="temp_chart")

with right:
    fig_hum = px.line(
        filtered,
        x="ts",
        y="humidity",
        title="Humidity Over Time"
    )
    st.plotly_chart(fig_hum, use_container_width=True, key="humidity_chart")

left2, right2 = st.columns(2)

with left2:
    fig_pred = px.scatter(
        pred_df,
        x="Actual_Temp",
        y="Predicted_RF",
        title="Actual vs Predicted Temperature (Random Forest)",
        opacity=0.4
    )
    st.plotly_chart(fig_pred, use_container_width=True, key="prediction_chart")

with right2:
    fig_models = px.bar(
        metrics_df,
        x="Model",
        y="R2",
        title="Model Comparison (R²)",
        text="R2"
    )
    st.plotly_chart(fig_models, use_container_width=True, key="model_chart")

# ---------------------------
# Dynamic chart
# ---------------------------
st.subheader(f"{sensor_option_label} Trend")
fig_dynamic = px.line(
    filtered,
    x="ts",
    y=sensor_option,
    title=f"{sensor_option_label} Over Time"
)
st.plotly_chart(fig_dynamic, use_container_width=True, key=f"dynamic_chart_{sensor_option}")

# ---------------------------
# Metrics table
# ---------------------------
st.subheader("Model Performance Metrics")
st.dataframe(metrics_df, use_container_width=True)

# ---------------------------
# Filtered data table
# ---------------------------
st.subheader("Filtered IoT Data")
st.dataframe(filtered.tail(100), use_container_width=True)

# ---------------------------
# Footer note
# ---------------------------
st.caption(
    "Note: Light and motion are binary sensor states. "
    "No Light / No Motion means no detection, while Light Detected / Motion Detected means the sensor detected activity."
)
