import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time
import requests
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
        color: #2c3e50 !important;
    }
    .alert-high-aqi {
        background-color: #ff0000;
        color: white;
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    with open('aqi_predictor_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv("city_day.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'AQI'])
    return df

# Function to fetch live AQI data
def get_live_aqi(city_name):
    url = f"https://api.waqi.info/feed/{city_name}/?token=fe0547e431226e44d33b4d50af849d737783f9de"
    try:
        response = requests.get(url)
        data = response.json()
        if data["status"] == "ok":
            aqi = data["data"]["aqi"]
            return aqi
        else:
            return None
    except:
        return None

model = load_model()
df = load_data()

# Enhanced AQI Recommendations Dictionary
aqi_recommendations = {
    'Good': """
        <b>Health</b>: Air quality is excellent, posing no health risks for anyone.  
        <b>Actions</b>: Enjoy outdoor activities like jogging, cycling, or family picnics without restrictions.  
        <b>Tips</b>:  
        - Take advantage of clean air to boost your physical and mental health through outdoor exercise.  
        - Open windows to naturally ventilate your home and improve indoor air quality.  
        - Support community initiatives like tree planting to maintain good air quality.  
        <b>Pro Tip</b>: Use this opportunity to promote eco-friendly habits, such as cycling or walking instead of driving, to keep the air clean!
    """,
    'Satisfactory': """
        <b>Health</b>: Air quality is acceptable, but sensitive groups (e.g., those with asthma or allergies) may experience mild discomfort.  
        <b>Actions</b>: Sensitive individuals should limit prolonged outdoor exertion, especially during midday when pollution peaks.  
        <b>Tips</b>:  
        - Use air purifiers with HEPA filters to keep indoor air clean and safe.  
        - Wear a mask (e.g., N95) if spending extended time outdoors.  
        - Stay updated with real-time air quality apps to plan outdoor activities.  
        <b>Pro Tip</b>: Incorporate air-purifying plants like peace lilies or snake plants indoors to enhance air quality naturally!
    """,
    'Moderate': """
        <b>Health</b>: Sensitive groups (e.g., children, elderly, or those with respiratory conditions) may experience health effects; the general public is less affected.  
        <b>Actions</b>: Sensitive groups should avoid strenuous outdoor activities; others should reduce prolonged exposure.  
        <b>Tips</b>:  
        - Wear N95 masks during outdoor activities to minimize pollutant inhalation.  
        - Schedule outdoor time for early morning or evening when pollution levels are typically lower.  
        - Use air quality monitoring apps to stay informed about real-time AQI changes.  
        <b>Pro Tip</b>: Invest in a high-quality air purifier for your home to create a safe indoor environment!
    """,
    'Poor': """
        <b>Health</b>: Everyone may experience health effects, with sensitive groups facing severe symptoms like coughing or breathing difficulties.  
        <b>Actions</b>: Avoid all outdoor activities, especially strenuous ones, and stay indoors with air purifiers running.  
        <b>Tips</b>:  
        - Seal windows and doors to block polluted air from entering your home.  
        - Use air purifiers with both HEPA and activated carbon filters for maximum pollutant removal.  
        - Stay hydrated and avoid indoor pollutant sources like incense or candles.  
        <b>Pro Tip</b>: Advocate for cleaner air by supporting local policies for reduced emissions, such as promoting public transport or green energy!
    """,
    'Very Poor': """
        <b>Health</b>: Serious health risks for everyone, including worsened respiratory and cardiovascular issues, even in healthy individuals.  
        <b>Actions</b>: Stay indoors at all times, avoiding any outdoor exposure, and keep air purifiers running continuously.  
        <b>Tips</b>:  
        - Monitor for symptoms like shortness of breath or chest pain and seek medical advice if needed.  
        - Use a humidifier to ease respiratory discomfort caused by dry air.  
        - Avoid indoor activities that generate pollutants, such as smoking or burning candles.  
        <b>Pro Tip</b>: Join community efforts to reduce pollution, like campaigns against vehicle idling or industrial emissions, to protect future air quality!
    """,
    'Severe': """
        <b>Health</b>: Emergency conditions with significant health risks for all, including severe respiratory and heart issues.  
        <b>Actions</b>: Remain indoors with all windows and doors sealed, avoiding any physical activity, indoors or out.  
        <b>Tips</b>:  
        - Use medical-grade air purifiers to ensure safe indoor air quality.  
        - Seek immediate medical help if you experience breathing difficulties or heart palpitations.  
        - Avoid cooking methods that produce smoke, like frying, to minimize indoor pollutants.  
        <b>Pro Tip</b>: Become an advocate for stricter air quality regulations to prevent severe AQI events and protect public health!
    """
}

# Long-term Consequences of Not Taking Precautions
long_term_consequences = """
<h3>🚨 Long-term Consequences of Ignoring Poor Air Quality (5+ Years)</h3>
Ignoring poor air quality can lead to serious consequences over time. Here’s what could happen if precautions are not taken:

- <b>Health Impacts</b>:  
  - <b>Chronic Respiratory Diseases</b>: Prolonged exposure to PM2.5, PM10, and NO2 can cause chronic obstructive pulmonary disease (COPD), asthma, and reduced lung function.  
  - <b>Cardiovascular Problems</b>: Pollutants entering the bloodstream increase the risk of heart attacks, strokes, and hypertension.  
  - <b>Cancer Risk</b>: Long-term exposure to PM2.5 is linked to lung cancer and other respiratory cancers.  
  - <b>Neurological Effects</b>: Emerging studies suggest air pollution may contribute to cognitive decline, dementia, and developmental issues in children.  
  - <b>Reduced Life Expectancy</b>: High AQI levels can shorten life expectancy by several years in heavily polluted areas.  

- <b>Environmental Impacts</b>:  
  - <b>Worsening Air Quality</b>: Without action, pollution levels will rise, leading to more frequent "Severe" AQI days.  
  - <b>Ecosystem Damage</b>: Pollutants like ozone harm vegetation, reducing crop yields and threatening food security.  
  - <b>Climate Feedback Loops</b>: Increased pollution worsens climate change, which exacerbates air quality issues through higher temperatures and stagnant air.  

- <b>Societal Impacts</b>:  
  - <b>Economic Costs</b>: Rising healthcare costs from pollution-related illnesses and lost productivity due to sick days.  
  - <b>Reduced Quality of Life</b>: Persistent poor air quality limits outdoor activities, impacting physical and mental well-being.  

<b>Take Action Now</b>: Support clean energy policies, use public transport, reduce personal emissions, and advocate for green spaces to improve air quality and prevent these long-term consequences.
"""

# Function to get AQI category based on numerical AQI value
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

# Function to get AQI category class for styling
def get_aqi_category_class(aqi_category):
    category_map = {
        'Good': 'aqi-good',
        'Satisfactory': 'aqi-moderate',
        'Moderate': 'aqi-unhealthy-sensitive',
        'Poor': 'aqi-unhealthy',
        'Very Poor': 'aqi-very-unhealthy',
        'Severe': 'aqi-hazardous'
    }
    return category_map.get(aqi_category, '')

# UI Title with emoji
st.title("🌿 Air Quality Index (AQI) Dashboard")
st.markdown("Explore air quality trends across cities with interactive visualizations, predictions, and live alerts.")

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
            "🏆 Top 10 Polluted Cities",
            "🚨 Live AQI Alerts"
        ],
        format_func=lambda x: x[2:]  # Remove emoji for cleaner selectbox
    )
    if st.button("Info about AQI"):
        st.header("Information about AQI and Parameters")
        st.markdown("""
            Air Quality Index (AQI) is a measure used to communicate how polluted the air currently is or how polluted it is forecast to become. AQI values are calculated based on the levels of key pollutants. Here’s an overview of the parameters used for prediction:

            - PM2.5 (Particulate Matter 2.5): Fine particles with a diameter of 2.5 micrometers or less. These can penetrate deep into the lungs and even enter the bloodstream, posing significant health risks.
            - PM10 (Particulate Matter 10): Coarser particles with a diameter of 10 micrometers or less. These can irritate the airways and lungs, especially in sensitive individuals.
            - NO2 (Nitrogen Dioxide): A gas produced by vehicle exhausts and industrial emissions. It can cause respiratory issues and contribute to the formation of smog.
            - CO (Carbon Monoxide): A colorless, odorless gas from incomplete combustion. High levels can lead to reduced oxygen delivery to the body’s organs and tissues.
            - O3 (Ozone): A gas formed by chemical reactions between oxides of nitrogen and volatile organic compounds. Ground-level ozone can cause breathing problems and damage lung tissue.

            These parameters are measured in units like µg/m³ (micrograms per cubic meter) or mg/m³ (milligrams per cubic meter) and are combined using a standardized formula to compute the AQI, which ranges from 0 (good) to 500 (hazardous).
        """)
    st.markdown("---")
    st.info("Select a view to explore AQI data, predict air quality, or check live alerts.")

# ---------------------------------------------
# 📊 Page 1: City-wise AQI Viewer
if page == "📊 City-wise AQI":
    st.header("📊 City-wise AQI Trends")
    col1, col2 = st.columns([3, 1])
    with col1:
        city = st.selectbox("Select a city", sorted(df['City'].unique()), key="city_select")
    
    # Filter data by the entire available dataset
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
# 🔮 Page 2: Predict AQI
elif page == "🔮 Predict AQI":
    st.header("🔮 AQI Prediction")
    st.markdown("Enter pollutant levels to predict the AQI value and category.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Pollutant Levels")
        pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 1000.0, 80.0, step=1.0)
        pm10 = st.number_input("PM10 (µg/m³)", 0.0, 1000.0, 100.0, step=1.0)
        no2 = st.number_input("NO2 (µg/m³)", 0.0, 500.0, 40.0, step=1.0)
        co = st.number_input("CO (mg/m³)", 0.0, 10.0, 1.0, step=0.1)
        o3 = st.number_input("O3 (µg/m³)", 0.0, 500.0, 30.0, step=1.0)
        
        # Predict button
        if st.button("Predict AQI", key="predict_button"):
            with st.spinner("Predicting..."):
                time.sleep(1)  # Simulate processing
                input_data = [pm25, pm10, no2, co, o3]
                predicted_aqi = model.predict([input_data])[0]
                aqi_category = get_aqi_category(predicted_aqi)
                st.write(f"Predicted AQI: {predicted_aqi:.2f}")
                st.markdown(f"<span class='{get_aqi_category_class(aqi_category)}'>Air Quality Category: <b>{aqi_category}</b></span>", unsafe_allow_html=True)
                st.balloons()
                # AQI Assistant response
                st.markdown("<h3>AQI Assistant</h3>", unsafe_allow_html=True)
                st.markdown(f"<div class='chatbot-message'>{aqi_recommendations.get(aqi_category, 'No recommendations available.')}</div>", unsafe_allow_html=True)
                # Long-term consequences
                st.markdown(long_term_consequences, unsafe_allow_html=True)

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
# 🏆 Page 5: Top 10 Polluted Cities
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

# ---------------------------------------------
# 🚨 Page 6: Live AQI Alerts
elif page == "🚨 Live AQI Alerts":
    st.header("🚨 Live AQI Alerts")
    st.markdown("Check real-time AQI for a selected city and receive alerts if air quality is poor.")

    city = st.selectbox("Select a city for live AQI", sorted(df['City'].unique()), key="live_city_select")
    
    with st.spinner(f"Fetching live AQI for {city}..."):
        live_aqi = get_live_aqi(city.lower())  # API expects lowercase city names
        if live_aqi is not None:
            aqi_category = get_aqi_category(live_aqi)
            st.write(f"Live AQI for {city}: {live_aqi:.2f}")
            st.markdown(f"<span class='{get_aqi_category_class(aqi_category)}'>Air Quality Category: <b>{aqi_category}</b></span>", unsafe_allow_html=True)
            
            # High AQI Alert
            if live_aqi > 100:  # Alert for Moderate or worse
                st.markdown(f"<div class='alert-high-aqi'>⚠️ High AQI Alert: Take precautions as air quality is {aqi_category.lower()}!</div>", unsafe_allow_html=True)
            
            # AQI Assistant Recommendations
            st.markdown("<h3>AQI Assistant</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='chatbot-message'>{aqi_recommendations.get(aqi_category, 'No recommendations available.')}</div>", unsafe_allow_html=True)
            
            # Long-term consequences
            st.markdown(long_term_consequences, unsafe_allow_html=True)
        else:
            st.error(f"Unable to fetch live AQI data for {city}. Please try another city or check your connection.")
