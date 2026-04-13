# 📊 IoT Sensor Data Prediction Dashboard

An interactive dashboard for analyzing IoT environmental sensor data and predicting temperature values using machine learning models.

---

## 🚀 Live Dashboard

👉 **Access the dashboard here:**  
[https://your-app-name.streamlit.app](https://iot-dashboard-project-fnu2xvwzpbq2ek9lvu9wrq.streamlit.app/)
---

## 📌 Project Description

This project analyzes IoT sensor data collected from environmental sensors, including temperature, humidity, gas levels, and motion detection.  

The goal is to:
- Explore sensor behavior over time
- Apply machine learning models to predict temperature
- Build an interactive dashboard for data visualization

---

## 📂 Dataset Features

The dataset includes the following sensors:

- **Temperature** → Measures air temperature  
- **Humidity** → Measures moisture level in the air  
- **CO Gas** → Measures carbon monoxide concentration  
- **LPG Gas** → Measures liquefied petroleum gas levels  
- **Smoke** → Measures smoke concentration  
- **Light** → Indicates if light is detected (0 = No, 1 = Yes)  
- **Motion** → Indicates if motion is detected (0 = No, 1 = Yes)  
- **Timestamp (ts)** → Time of sensor reading  

---

## 🧠 Machine Learning Models

We implemented and compared the following models:

- Linear Regression  
- Random Forest Regressor  

### 📊 Evaluation Metric:
- **R² Score**

The best-performing model was selected based on the highest R² value.

---

## 📈 Dashboard Features

The dashboard provides:

- 📅 Date range filtering  
- ⏰ Hour-based filtering  
- 🌡️ Sensor selection  
- 💡 Light condition filtering  
- 🚶 Motion detection filtering  

### Visualizations:
- Temperature over time  
- Humidity over time  
- Actual vs Predicted values  
- Model comparison (R²)  
- Dynamic sensor trends  

---

## 🛠️ Technologies Used

- Python  
- Streamlit  
- Pandas  
- Plotly  
- Scikit-learn  

---

## 📁 Project Structure
- app.py
- clean_iot_data_small.csv
- predictions.csv
- metrics.csv
- requirements.txt
- README.md

  
---

## ⚠️ Notes

- A sample of the dataset was used in the dashboard to ensure performance and faster loading.
- Light and motion are binary features:
  - 0 → Not detected  
  - 1 → Detected  

---

## 🎯 Conclusion

This project demonstrates how IoT data can be combined with machine learning to generate meaningful insights and predictions, presented through an interactive dashboard for better user experience.

---
