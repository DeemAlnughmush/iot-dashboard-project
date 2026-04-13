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
    "Interactive dashboard for analyzing IoT sensor trends and "
    "temperature prediction using machine learning."
)

# ---------------------------
# Sidebar filters
# ---------------------------
st.sidebar.header("Filters")

min_date = df["ts"].min().date()
max_date = df["ts"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    [min_date, max_date]
)

sensor_option = st.sidebar.selectbox(
    "Select sensor",
    ["temp", "humidity", "co", "lpg", "smoke"]
)

hour_range = st.sidebar.slider(
    "Hour range",
    0, 23, (0, 23)
)

light_option = st.sidebar.selectbox("Light", ["All", 0, 1])
motion_option = st.sidebar.selectbox("Motion", ["All", 0, 1])

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

if light_option != "All":
    filtered = filtered[filtered["light"] == light_option]

if motion_option != "All":
    filtered = filtered[filtered["motion"] == motion_option]

# ---------------------------
# KPI Cards
# ---------------------------
best_model_row = metrics_df.loc[metrics_df["R2"].idxmax()]
best_model = best_model_row["Model"]
best_r2 = best_model_row["R2"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Average Temp", round(filtered["temp"].mean(), 3))
c2.metric("Average Humidity", round(filtered["humidity"].mean(), 3))
c3.metric("Average Smoke", round(filtered["smoke"].mean(), 3))
c4.metric("Best Model R²", f"{best_model} ({best_r2:.3f})")

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
    st.plotly_chart(fig_temp, use_container_width=True)

with right:
    fig_hum = px.line(
        filtered,
        x="ts",
        y="humidity",
        title="Humidity Over Time"
    )
    st.plotly_chart(fig_hum, use_container_width=True)

left2, right2 = st.columns(2)

with left2:
    fig_pred = px.scatter(
        pred_df,
        x="Actual_Temp",
        y="Predicted_RF",
        title="Actual vs Predicted Temperature (Random Forest)",
        opacity=0.4
    )
    st.plotly_chart(fig_pred, use_container_width=True)

with right2:
    fig_models = px.bar(
        metrics_df,
        x="Model",
        y="R2",
        title="Model Comparison (R²)",
        text="R2"
    )
    st.plotly_chart(fig_models, use_container_width=True)

# ---------------------------
# Dynamic chart
# ---------------------------
st.subheader(f"{sensor_option.capitalize()} Trend")
fig_dynamic = px.line(
    filtered,
    x="ts",
    y=sensor_option,
    title=f"{sensor_option.capitalize()} Over Time"
)
st.plotly_chart(fig_dynamic, use_container_width=True)

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
