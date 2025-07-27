import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
from io import StringIO

# Set page configuration with a custom theme
st.set_page_config(
    page_title="Air Quality Index Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
        padding: 20px;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    .stSelectbox, .stDateInput {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 5px;
    }
    .aqi-good { background-color: #00e400; color: white; padding: 5px; border-radius: 5px; }
    .aqi-moderate { background-color: #ffff00; color: black; padding: 5px; border-radius: 5px; }
    .aqi-unhealthy-sensitive { background-color: #ff7e00; color: white; padding: 5px; border-radius: 5px; }
    .aqi-unhealthy { background-color: #ff0000; color: white; padding: 5px; border-radius: 5px; }
    .aqi-very-unhealthy { background-color: #8f3f97; color: white; padding: 5px; border-radius: 5px; }
    .aqi-hazardous { background-color: #7e0023; color: white; padding: 5px; border-radius: 5px; }
    .chatbot-message {
        background-color: #f3e8ff !important;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        border-left: 5px solid #7c3aed !important;
        font-family: 'Arial', sans-serif;
        color: #2c3e50 !important; /* Ensure text is visible */
    }
    </style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    return joblib.load('aqi_predictor_model.pkl')

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv("city_day.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'AQI'])
    return df

model = load_model()
df = load_data()

# AQI Recommendations Dictionary
aqi_recommendations = {
    'Good': """
        Health: Air quality is satisfactory, and air pollution poses little or no risk.
        Actions: Enjoy outdoor activities without restrictions.
        Tips: Continue monitoring air quality for any unexpected changes.
    """,
    'Moderate': """
        Health: Air quality is acceptable; however, some pollutants may affect sensitive individuals.
        Actions: Sensitive groups (e.g., those with respiratory issues) should reduce prolonged outdoor exertion.
        Tips: Use air purifiers indoors and keep windows closed during peak pollution hours.
    """,
    'Unhealthy for Sensitive Groups': """
         Health: Members of sensitive groups may experience health effects; the general public is less likely to be affected.
         Actions: Sensitive groups should avoid prolonged outdoor activities. Others should limit exertion.
         Tips: Wear masks (e.g., N95) outdoors and ensure good indoor ventilation.
    """,
    'Unhealthy': """
         Health: Everyone may begin to experience health effects; sensitive groups face more serious effects.
         Actions: Avoid outdoor activities, especially strenuous ones. Stay indoors with air purifiers.
         Tips: Seal windows and doors, and use HEPA filters to reduce indoor pollutants.
    """,
    'Very Unhealthy': """
         Health: Health alert: everyone may experience serious health effects.
        Actions: Stay indoors and avoid all outdoor activities. Use air purifiers continuously.
        Tips: Monitor health symptoms and seek medical advice if respiratory issues arise.
    """,
    'Hazardous': """
        Health: Emergency conditions; the entire population is likely to be affected.
        Actions: Remain indoors with all windows and doors sealed. Avoid any physical activity.
        Tips: Use high-quality air purifiers and consult a doctor for any breathing difficulties.
    """
}

# Function to get AQI category class for styling
def get_aqi_category_class(aqi_category):
    category_map = {
        'Good': 'aqi-good',
        'Moderate': 'aqi-moderate',
        'Unhealthy for Sensitive Groups': 'aqi-unhealthy-sensitive',
        'Unhealthy': 'aqi-unhealthy',
        'Very Unhealthy': 'aqi-very-unhealthy',
        'Hazardous': 'aqi-hazardous'
    }
    return category_map.get(aqi_category, '')

# UI Title with emoji
st.title("🌿 Air Quality Index (AQI) Dashboard")
st.markdown("Explore air quality trends across cities with interactive visualizations, predictions, and insights.")

# Sidebar with enhanced styling
with st.sidebar:
    st.header("Navigation")
    page = st.selectbox(
        "Choose View",
        [
            "📊 City-wise AQI",
            "🔮 Predict AQI",
            "🆚 Compare Cities",
            "🔥 Heatmap",
            "🏆 Top 10 Polluted Cities"
        ],
        format_func=lambda x: x[2:]  # Remove emoji for cleaner selectbox
    )
    if st.button("Info about AQI"):
        st.header("Information about AQI and Parameters")
        st.markdown("""
            **Air Quality Index (AQI)** is a measure used to communicate how polluted the air currently is or how polluted it is forecast to become. AQI values are calculated based on the levels of key pollutants. Here’s an overview of the parameters used for prediction:

            - **PM2.5 (Particulate Matter 2.5)**: Fine particles with a diameter of 2.5 micrometers or less. These can penetrate deep into the lungs and even enter the bloodstream, posing significant health risks.
            - **PM10 (Particulate Matter 10)**: Coarser particles with a diameter of 10 micrometers or less. These can irritate the airways and lungs, especially in sensitive individuals.
            - **NO2 (Nitrogen Dioxide)**: A gas produced by vehicle exhausts and industrial emissions. It can cause respiratory issues and contribute to the formation of smog.
            - **CO (Carbon Monoxide)**: A colorless, odorless gas from incomplete combustion. High levels can lead to reduced oxygen delivery to the body’s organs and tissues.
            - **O3 (Ozone)**: A gas formed by chemical reactions between oxides of nitrogen and volatile organic compounds. Ground-level ozone can cause breathing problems and damage lung tissue.

            These parameters are measured in units like µg/m³ (micrograms per cubic meter) or mg/m³ (milligrams per cubic meter) and are combined using a standardized formula to compute the AQI, which ranges from 0 (good) to 500 (hazardous).
        """)
    st.markdown("---")
    st.info("Select a view to explore AQI data or predict future air quality.")

# ---------------------------------------------
# 📊 Page 1: City-wise AQI Viewer
if page == "📊 City-wise AQI":
    st.header("📊 City-wise AQI Trends")
    col1, col2 = st.columns([3, 1])
    with col1:
        city = st.selectbox("Select a city", sorted(df['City'].unique()), key="city_select")
    
    # Filter data by the entire available dataset (no date range)
    city_df = df[df['City'] == city]
    
    with st.spinner("Loading AQI trend..."):
        st.line_chart(
            city_df.set_index('Date')['AQI'],
            height=400,
            use_container_width=True
        )
    
    # Summary Table
    st.subheader("AQI Summary Statistics")
    summary = city_df['AQI'].agg(['mean', 'min', 'max']).to_frame().T
    summary.columns = ['Mean AQI', 'Min AQI', 'Max AQI']
    st.table(summary.round(2))
    
    # Pollutant Contribution Pie Chart
    st.subheader("Pollutant Contribution")
    pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3']
    pollutant_means = city_df[pollutants].mean()
    fig, ax = plt.subplots()
    ax.pie(pollutant_means, labels=pollutants, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
    
    # Download Button
    csv = city_df.to_csv(index=False)
    st.download_button(
        label="Download City Data as CSV",
        data=csv,
        file_name=f"{city}_aqi_data.csv",
        mime="text/csv"
    )
    st.markdown(f"Trend Analysis for {city}: Visualize AQI fluctuations and pollutant contributions.")

# ---------------------------------------------
# 🔮 Page 2: Predict AQI Category
elif page == "🔮 Predict AQI":
    st.header("🔮 AQI Prediction")
    st.markdown("Enter pollutant levels to predict the AQI category.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Pollutant Levels")
        pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 1000.0, 80.0, step=1.0)
        pm10 = st.number_input("PM10 (µg/m³)", 0.0, 1000.0, 100.0, step=1.0)
        no2 = st.number_input("NO2 (µg/m³)", 0.0, 500.0, 40.0, step=1.0)
        co = st.number_input("CO (mg/m³)", 0.0, 10.0, 1.0, step=0.1)
        o3 = st.number_input("O3 (µg/m³)", 0.0, 500.0, 30.0, step=1.0)
        
        # Predict button
        if st.button("Predict AQI Category", key="predict_button"):
            with st.spinner("Predicting..."):
                time.sleep(1)  # Simulate processing
                result = model.predict([[pm25, pm10, no2, co, o3]])
                aqi_category = result[0]
                st.markdown(f"<span class='{get_aqi_category_class(aqi_category)}'>Predicted AQI Category: {aqi_category}</span>", unsafe_allow_html=True)
                st.balloons()
                # Chatbot response
                st.markdown("AQI Assistant:")
                st.markdown(f"<div class='chatbot-message'>{aqi_recommendations.get(aqi_category, 'No recommendations available.')}</div>", unsafe_allow_html=True)

# ---------------------------------------------
# 🆚 Page 3: Compare Two Cities
elif page == "🆚 Compare Cities":
    st.header("🆚 Compare AQI Between Cities")
    cities = sorted(df['City'].unique())
    
    col1, col2 = st.columns(2)
    with col1:
        city1 = st.selectbox("City 1", cities, index=0, key="city1_select")
    with col2:
        city2 = st.selectbox("City 2", cities, index=1, key="city2_select")
    
    # Filter data without date range
    city1_df = df[df['City'] == city1]
    city2_df = df[df['City'] == city2]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"AQI Trend: {city1}")
        with st.spinner(f"Loading {city1} data..."):
            st.line_chart(
                city1_df.set_index('Date')['AQI'],
                height=350,
                use_container_width=True
            )
    
    with col2:
        st.subheader(f"AQI Trend: {city2}")
        with st.spinner(f"Loading {city2} data..."):
            st.line_chart(
                city2_df.set_index('Date')['AQI'],
                height=350,
                use_container_width=True
            )
    
    # Download Button for Comparison Data
    combined_df = pd.concat([city1_df, city2_df])
    csv = combined_df.to_csv(index=False)
    st.download_button(
        label="Download Comparison Data as CSV",
        data=csv,
        file_name=f"{city1}_vs_{city2}_aqi_data.csv",
        mime="text/csv"
    )

# ---------------------------------------------
# 🔥 Page 4: Heatmap of AQI by Month and City
elif page == "🔥 Heatmap":
    st.header("🔥 AQI Heatmap by Month and City")
    with st.spinner("Generating heatmap..."):
        heat_df = df.copy()
        heat_df['Month'] = heat_df['Date'].dt.month
        pivot = heat_df.pivot_table(index='City', columns='Month', values='AQI', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(pivot, cmap="YlOrRd", ax=ax, annot=True, fmt=".1f", cbar_kws={'label': 'AQI'})
        ax.set_title("Average AQI by City and Month")
        st.pyplot(fig)
    st.markdown("Insight: Red indicates higher AQI (worse air quality). Compare monthly patterns across cities.")

# ---------------------------------------------
# 🏆 Page 5: Top 10 Most Polluted Cities
elif page == "🏆 Top 10 Polluted Cities":
    st.header("🏆 Top 10 Most Polluted Cities")
    with st.spinner("Calculating rankings..."):
        avg_aqi = df.groupby('City')['AQI'].mean().sort_values(ascending=False).head(10)
        st.bar_chart(
            avg_aqi,
            height=400,
            use_container_width=True
        )
    st.markdown("Insight: These cities have the highest average AQI, indicating poorer air quality.")