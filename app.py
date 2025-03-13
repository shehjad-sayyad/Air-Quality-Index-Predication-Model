import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ✅ Ensure the first Streamlit command
st.set_page_config(page_title="AQI Prediction", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("aqi_model.pkl")

model = load_model()

# Load dataset for visualizations
@st.cache_data
def load_data():
    df = pd.read_csv("air quality data set.csv")
    return df

df = load_data()

# --- Streamlit UI ---
st.title("🌍 AQI Prediction Model")
st.markdown("### Know Your Air Quality & Stay Safe!")

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Predict AQI", "Data Visualizations", "Model Insights"])

# --- AQI Prediction Page ---
if page == "Predict AQI":
    st.header("📊 Predict Air Quality Index")

    col1, col2 = st.columns(2)
    with col1:
        pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 500.0, 30.0)
        pm10 = st.number_input("PM10 (µg/m³)", 0.0, 500.0, 50.0)
        no = st.number_input("NO (ppb)", 0.0, 200.0, 10.0)
        no2 = st.number_input("NO2 (ppb)", 0.0, 200.0, 20.0)
        nox = st.number_input("NOx (ppb)", 0.0, 200.0, 25.0)
        nh3 = st.number_input("NH3 (ppb)", 0.0, 200.0, 5.0)
    
    with col2:
        co = st.number_input("CO (ppm)", 0.0, 10.0, 1.0)
        so2 = st.number_input("SO2 (ppb)", 0.0, 200.0, 15.0)
        o3 = st.number_input("O3 (ppb)", 0.0, 200.0, 30.0)
        benzene = st.number_input("Benzene (µg/m³)", 0.0, 50.0, 1.0)
        toluene = st.number_input("Toluene (µg/m³)", 0.0, 200.0, 5.0)
        xylene = st.number_input("Xylene (µg/m³)", 0.0, 200.0, 2.0)

    if st.button("🔍 Predict AQI"):
        # Prepare input data as DataFrame
        input_data = pd.DataFrame([[pm25, pm10, no, no2, nox, nh3, co, so2, o3, benzene, toluene, xylene]], 
                                  columns=['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 
                                           'Benzene', 'Toluene', 'Xylene'])
        
        # Predict AQI
        prediction = model.predict(input_data)[0]
        
        # AQI Category
        def get_aqi_category(aqi):
            if aqi <= 50:
                return "Good (Green)"
            elif aqi <= 100:
                return "Moderate (Yellow)"
            elif aqi <= 150:
                return "Unhealthy for Sensitive Groups (Orange)"
            elif aqi <= 200:
                return "Unhealthy (Red)"
            elif aqi <= 300:
                return "Very Unhealthy (Purple)"
            else:
                return "Hazardous (Maroon)"

        category = get_aqi_category(prediction)
        
        # Display Result
        st.success(f"Predicted AQI: {round(prediction, 2)}")
        st.markdown(f"**Category:** {category}")

# --- Data Visualization Page ---
elif page == "Data Visualizations":
    st.header("📊 Air Quality Data Visualizations")

    # Plot AQI Distribution
    st.subheader("AQI Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["AQI"], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    # Heatmap of Correlations
    st.subheader("Correlation Between Pollutants")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # AQI Levels by City (Example if dataset contains locations)
    if "City" in df.columns:
        st.subheader("AQI Levels by City")
        fig = px.bar(df, x="City", y="AQI", color="AQI", title="AQI Levels by City")
        st.plotly_chart(fig)

# --- Model Insights Page ---
elif page == "Model Insights":
    st.header("🛠 Machine Learning Model Insights")

    # Feature Importance (Random Forest Example)
    st.subheader("Feature Importance")
    feature_importance = model.feature_importances_
    feature_names = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    
    # Plot Feature Importance
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=feature_importance, y=feature_names, palette="Blues", ax=ax)
    ax.set_title("Feature Importance in AQI Prediction")
    st.pyplot(fig)

    # Model Comparison (Assuming multiple models were trained)
    st.subheader("Model Comparison")
    model_results = pd.DataFrame({
        "Model": ["Random Forest", "Decision Tree", "Linear Regression"],
        "Accuracy": [92, 85, 78]  # Example accuracy scores
    })

    fig = px.bar(model_results, x="Model", y="Accuracy", color="Accuracy", title="Model Accuracy Comparison")
    st.plotly_chart(fig)

    st.markdown("Based on the accuracy scores, **Random Forest** is the best model for predicting AQI.")

# --- Footer ---
st.sidebar.markdown("Developed with ❤️ using Streamlit")
