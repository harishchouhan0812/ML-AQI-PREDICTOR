import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time
import requests
from io import StringIO
import numpy as np

st.set_page_config(
    page_title="Air Quality Index Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .stSelectbox, .stDateInput, .stSlider {
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
    .assistant-response {
        background-color: #e6f3ff;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        border-left: 5px solid #1e90ff;
        font-family: 'Arial', sans-serif;
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open('aqi_predictor_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    df = pd.read_csv("city_day.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.strftime('%b')  # Extract month abbreviation
    df = df.dropna(subset=['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'AQI'])
    all_cities = sorted(df['City'].unique())
    return df, all_cities

def get_live_aqi(city_name):
    url = f"https://api.waqi.info/feed/{city_name}/?token=fe0547e431226e44d33b4d50af849d737783f9de"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data["status"] == "ok":
            aqi = data["data"]["aqi"]
            return aqi
        else:
            return None
    except:
        return None

def get_weather_data(city_name):
    api_key = "your_openweathermap_api_key"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data["cod"] == 200:
            return {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"]
            }
        else:
            return None
    except:
        return None

def estimate_tree_impact(num_trees, current_aqi):
    pm25_reduction = num_trees * 0.3 / 10000
    co2_reduction = num_trees * 0.022
    aqi_reduction = pm25_reduction * 2
    new_aqi = max(0, current_aqi - aqi_reduction)
    return {
        "new_aqi": new_aqi,
        "pm25_reduction": pm25_reduction,
        "co2_reduction": co2_reduction,
        "category": get_aqi_category(new_aqi)
    }

def estimate_car_removal_impact(num_cars, current_aqi):
    pm25_reduction = num_cars * 0.3 / 1000
    co2_reduction = num_cars * 4.6
    no2_reduction = num_cars * 0.1 / 1000
    aqi_reduction = (pm25_reduction * 2 + no2_reduction * 1.5)
    new_aqi = max(0, current_aqi - aqi_reduction)
    return {
        "new_aqi": new_aqi,
        "pm25_reduction": pm25_reduction,
        "co2_reduction": co2_reduction,
        "no2_reduction": no2_reduction,
        "category": get_aqi_category(new_aqi)
    }

model = load_model()
df, all_cities = load_data()

aqi_recommendations = {
    'Good': {
        'General': """
            <b>Health</b>: Air quality is excellent, posing no health risks for anyone.  
            <b>Actions</b>: Enjoy outdoor activities like jogging, cycling, or family picnics without restrictions.  
            <b>Tips</b>:  
            - Take advantage of clean air to boost physical and mental health through outdoor exercise.  
            - Open windows to naturally ventilate your home and improve indoor air quality.  
            - Support community initiatives like tree planting to maintain good air quality.  
            <b>Pro Tip</b>: Use this opportunity to promote eco-friendly habits, such as cycling or walking instead of driving, to keep the air clean!
        """,
        'Asthma': """
            <b>For Asthma Patients</b>: The air is safe for you today! Enjoy outdoor activities, but keep your inhaler handy as a precaution.  
            <b>Tips</b>:  
            - Engage in light exercise like walking to improve lung capacity, but avoid overexertion.  
            - Ensure your home is dust-free to complement the clean outdoor air.  
            - Monitor pollen levels, as they can still trigger symptoms even in good AQI conditions.  
            <b>Pro Tip</b>: Practice breathing exercises like diaphragmatic breathing to strengthen respiratory health.
        """,
        'Heart Disease': """
            <b>For Heart Disease Patients</b>: The clean air is ideal for light outdoor activities to support heart health.  
            <b>Tips</b>:  
            - Try gentle activities like walking or yoga to improve circulation without straining your heart.  
            - Stay hydrated and avoid extreme temperatures, even with good air quality.  
            - Keep medications accessible in case of unexpected symptoms.  
            <b>Pro Tip</b>: Join a community walking group to stay motivated and socially connected while staying active.
        """,
        'Elderly': """
            <b>For the Elderly</b>: The air quality is perfect for enjoying outdoor time safely.  
            <b>Tips</b>:  
            - Take short walks or engage in light gardening to stay active and boost mood.  
            - Ensure good hydration and rest to maintain energy levels.  
            - Avoid crowded areas to reduce exposure to minor pollutants or allergens.  
            <b>Pro Tip</b>: Spend time in green spaces like parks to enhance mental well-being.
        """,
        'Children': """
            <b>For Children</b>: Great air quality for kids to play outside and stay active!  
            <b>Tips</b>:  
            - Encourage outdoor games like tag or soccer to promote physical development.  
            - Ensure kids stay hydrated and take breaks to avoid fatigue.  
            - Apply sunscreen and monitor for allergens like pollen that may still be present.  
            <b>Pro Tip</b>: Organize outdoor playdates to foster social skills in a healthy environment.
        """,
        'Pregnancy': """
            <b>For Pregnant Individuals</b>: The air is safe for you and your baby to enjoy outdoor activities.  
            <b>Tips</b>:  
            - Take gentle walks to support circulation and reduce pregnancy-related discomfort.  
            - Avoid overexertion and stay hydrated to maintain energy levels.  
            - Monitor for any unusual symptoms and consult your doctor if needed.  
            <b>Pro Tip</b>: Practice prenatal yoga outdoors to combine fresh air with relaxation.
        """
    },
    'Satisfactory': {
        'General': """
            <b>Health</b>: Air quality is acceptable, but sensitive groups may experience mild discomfort.  
            <b>Actions</b>: Sensitive individuals should limit prolonged outdoor exertion, especially during midday when pollution peaks.  
            <b>Tips</b>:  
            - Use air purifiers with HEPA filters to keep indoor air clean.  
            - Wear a mask (e.g., N95) if spending extended time outdoors.  
            - Stay updated with real-time air quality apps to plan outdoor activities.  
            <b>Pro Tip</b>: Incorporate air-purifying plants like peace lilies or snake plants indoors to enhance air quality naturally!
        """,
        'Asthma': """
            <b>For Asthma Patients</b>: Be cautious, as mild pollutants may trigger symptoms.  
            <b>Tips</b>:  
            - Limit outdoor time to early morning or evening when pollution is lower.  
            - Use your inhaler proactively before any outdoor activity.  
            - Keep windows closed and use an air purifier to reduce indoor irritants.  
            <b>Pro Tip</b>: Avoid areas with heavy traffic to minimize exposure to vehicle emissions.
        """,
        'Heart Disease': """
            <b>For Heart Disease Patients</b>: Mild air pollution may stress your cardiovascular system.  
            <b>Tips</b>:  
            - Avoid strenuous outdoor activities; opt for indoor exercises like stretching.  
            - Monitor heart rate and blood pressure, and keep medications nearby.  
            - Use a HEPA air purifier at home to maintain clean indoor air.  
            <b>Pro Tip</b>: Practice stress-relief techniques like meditation to support heart health in suboptimal air conditions.
        """,
        'Elderly': """
            <b>For the Elderly</b>: Air quality may cause minor discomfort, especially with prolonged exposure.  
            <b>Tips</b>:  
            - Limit outdoor activities to short durations and avoid peak pollution hours.  
            - Stay indoors with good ventilation and use air purifiers.  
            - Monitor for fatigue or respiratory discomfort and rest as needed.  
            <b>Pro Tip</b>: Engage in indoor hobbies like reading or puzzles to stay active without exposure.
        """,
        'Children': """
            <b>For Children</b>: Kids can play outside, but limit intense activities to avoid irritation.  
            <b>Tips</b>:  
            - Schedule outdoor play in the early morning or late afternoon.  
            - Ensure kids wear masks during high-energy activities to reduce pollutant inhalation.  
            - Keep indoor spaces clean and free of dust or allergens.  
            <b>Pro Tip</b>: Encourage indoor activities like board games on days with higher pollution.
        """,
        'Pregnancy': """
            <b>For Pregnant Individuals</b>: Air quality is mostly safe but may cause mild discomfort.  
            <b>Tips</b>:  
            - Avoid long outdoor walks, especially near busy roads or industrial areas.  
            - Use a HEPA air purifier at home to ensure clean indoor air.  
            - Stay hydrated and rest frequently to support your health.  
            <b>Pro Tip</b>: Practice indoor prenatal exercises to stay active safely.
        """
    },
    'Moderate': {
        'General': """
            <b>Health</b>: Sensitive groups may experience health effects; the general public is less affected.  
            <b>Actions</b>: Sensitive groups should avoid strenuous outdoor activities; others should reduce prolonged exposure.  
            <b>Tips</b>:  
            - Wear N95 masks during outdoor activities to minimize pollutant inhalation.  
            - Schedule outdoor time for early morning or evening when pollution levels are lower.  
            - Use air quality monitoring apps to stay informed about real-time AQI changes.  
            <b>Pro Tip</b>: Invest in a high-quality air purifier for your home to create a safe indoor environment!
        """,
        'Asthma': """
            <b>For Asthma Patients</b>: Moderate air quality increases the risk of asthma attacks.  
            <b>Tips</b>:  
            - Avoid outdoor exercise; use indoor alternatives like treadmill walking.  
            - Keep rescue inhalers accessible and use as prescribed before any exposure.  
            - Use a HEPA air purifier and keep windows closed to reduce indoor pollutants.  
            <b>Pro Tip</b>: Track your symptoms daily and consult your doctor if you notice increased wheezing or shortness of breath.
        """,
        'Heart Disease': """
            <b>For Heart Disease Patients</b>: Pollutants may strain your heart, increasing symptom risk.  
            <b>Tips</b>:  
            - Stay indoors and avoid physical exertion to prevent cardiovascular stress.  
            - Monitor for symptoms like chest pain or palpitations and seek medical help if needed.  
            - Use air purifiers with activated carbon filters to remove gases like NO2.  
            <b>Pro Tip</b>: Maintain a heart-healthy diet rich in antioxidants to counter pollutant effects.
        """,
        'Elderly': """
            <b>For the Elderly</b>: Increased risk of respiratory and cardiovascular issues in moderate air quality.  
            <b>Tips</b>:  
            - Stay indoors during peak pollution hours (midday) and use air purifiers.  
            - Monitor for symptoms like shortness of breath or fatigue and rest as needed.  
            - Keep medications handy and consult a doctor if symptoms worsen.  
            <b>Pro Tip</b>: Stay hydrated and practice gentle indoor exercises like chair yoga to maintain mobility.
        """,
        'Children': """
            <b>For Children</b>: Moderate air quality may affect developing lungs.  
            <b>Tips</b>:  
            - Limit outdoor playtime and avoid high-energy activities like running.  
            - Ensure kids wear N95 masks if they must go outside.  
            - Use air purifiers in bedrooms and play areas to maintain clean air.  
            <b>Pro Tip</b>: Engage kids in educational indoor activities like science kits to keep them occupied safely.
        """,
        'Pregnancy': """
            <b>For Pregnant Individuals</b>: Moderate pollution may pose risks to you and your baby.  
            <b>Tips</b>:  
            - Avoid outdoor activities, especially in areas with heavy traffic or industry.  
            - Use a medical-grade air purifier to ensure clean indoor air.  
            - Monitor for symptoms like fatigue or respiratory discomfort and consult your doctor.  
            <b>Pro Tip</b>: Rest in well-ventilated, clean indoor spaces to support a healthy pregnancy.
        """
    },
    'Poor': {
        'General': """
            <b>Health</b>: Everyone may experience health effects, with sensitive groups facing severe symptoms.  
            <b>Actions</b>: Avoid all outdoor activities and stay indoors with air purifiers running.  
            <b>Tips</b>:  
            - Seal windows and doors to block polluted air from entering your home.  
            - Use air purifiers with both HEPA and activated carbon filters for maximum pollutant removal.  
            - Stay hydrated and avoid indoor pollutant sources like incense or candles.  
            <b>Pro Tip</b>: Advocate for cleaner air by supporting local policies for reduced emissions!
        """,
        'Asthma': """
            <b>For Asthma Patients</b>: Poor air quality significantly increases the risk of severe asthma attacks.  
            <b>Tips</b>:  
            - Stay indoors and avoid any outdoor exposure, even for short periods.  
            - Use a nebulizer or rescue inhaler as prescribed and keep it accessible.  
            - Run a medical-grade air purifier continuously to minimize indoor pollutants.  
            <b>Pro Tip</b>: Contact your doctor immediately if you experience frequent symptoms or worsening attacks.
        """,
        'Heart Disease': """
            <b>For Heart Disease Patients</b>: Poor air quality can trigger serious cardiovascular events.  
            <b>Tips</b>:  
            - Remain indoors with sealed windows and use a high-quality air purifier.  
            - Monitor heart rate and blood pressure closely; seek medical help for any chest pain.  
            - Avoid stress or physical exertion to reduce strain on your heart.  
            <b>Pro Tip</b>: Keep emergency contacts and medications readily available in case of sudden symptoms.
        """,
        'Elderly': """
            <b>For the Elderly</b>: Poor air quality poses a high risk of respiratory and heart issues.  
            <b>Tips</b>:  
            - Stay indoors with air purifiers running and avoid any physical activity.  
            - Monitor for symptoms like coughing, dizziness, or chest tightness; seek medical advice if needed.  
            - Ensure a comfortable, clean indoor environment with proper humidity.  
            <b>Pro Tip</b>: Use telehealth services to consult doctors without leaving home.
        """,
        'Children': """
            <b>For Children</b>: Poor air quality can harm developing lungs and overall health.  
            <b>Tips</b>:  
            - Keep children indoors and use air purifiers in living and sleeping areas.  
            - Avoid activities that increase breathing rates, even indoors.  
            - Monitor for coughing or wheezing and consult a pediatrician if symptoms appear.  
            <b>Pro Tip</b>: Create a clean, fun indoor environment with games or crafts to keep kids engaged.
        """,
        'Pregnancy': """
            <b>For Pregnant Individuals</b>: Poor air quality may affect fetal development and maternal health.  
            <b>Tips</b>:  
            - Stay indoors with sealed windows and use a medical-grade air purifier.  
            - Avoid any physical exertion and monitor for symptoms like shortness of breath.  
            - Consult your obstetrician if you feel unwell or notice reduced fetal movement.  
            <b>Pro Tip</b>: Rest in a calm, clean indoor space to support a healthy pregnancy.
        """
    },
    'Very Poor': {
        'General': """
            <b>Health</b>: Serious health risks for everyone, including worsened respiratory and cardiovascular issues.  
            <b>Actions</b>: Stay indoors at all times, avoiding any outdoor exposure, and keep air purifiers running continuously.  
            <b>Tips</b>:  
            - Monitor for symptoms like shortness of breath or chest pain and seek medical advice if needed.  
            - Use a humidifier to ease respiratory discomfort caused by dry air.  
            - Avoid indoor activities that generate pollutants, such as smoking or burning candles.  
            <b>Pro Tip</b>: Join community efforts to reduce pollution, like campaigns against vehicle idling!
        """,
        'Asthma': """
            <b>For Asthma Patients</b>: Very poor air quality can trigger life-threatening asthma attacks.  
            <b>Tips</b>:  
            - Remain indoors with a medical-grade air purifier running at all times.  
            - Use preventive medications and keep emergency inhalers or nebulizers ready.  
            - Avoid any indoor irritants like dust, pet dander, or strong cleaning products.  
            <b>Pro Tip</b>: Have an asthma action plan and emergency contacts ready for immediate response.
        """,
        'Heart Disease': """
            <b>For Heart Disease Patients</b>: Very poor air quality significantly increases the risk of heart attacks or arrhythmias.  
            <b>Tips</b>:  
            - Stay indoors with sealed windows and use air purifiers with HEPA and carbon filters.  
            - Avoid any physical or emotional stress; monitor for chest pain or irregular heartbeats.  
            - Keep emergency medications like nitroglycerin accessible and contact a doctor if symptoms arise.  
            <b>Pro Tip</b>: Use a pulse oximeter to monitor oxygen levels and heart rate at home.
        """,
        'Elderly': """
            <b>For the Elderly</b>: Very poor air quality poses severe risks to respiratory and cardiovascular health.  
            <b>Tips</b>:  
            - Stay indoors with air purifiers and avoid any physical activity, even indoors.  
            - Monitor for symptoms like confusion, shortness of breath, or chest pain; seek immediate help if needed.  
            - Maintain proper hydration and a comfortable indoor temperature.  
            <b>Pro Tip</b>: Arrange for regular check-ins with family or caregivers during high-pollution days.
        """,
        'Children': """
            <b>For Children</b>: Very poor air quality can cause serious harm to developing lungs and immune systems.  
            <b>Tips</b>:  
            - Keep children indoors with air purifiers running in all living spaces.  
            - Avoid any physical activities that increase breathing rates.  
            - Watch for signs of respiratory distress and consult a pediatrician immediately if noticed.  
            <b>Pro Tip</b>: Use engaging indoor activities like storytelling or educational apps to keep kids calm and safe.
        """,
        'Pregnancy': """
            <b>For Pregnant Individuals</b>: Very poor air quality poses significant risks to both maternal and fetal health.  
            <b>Tips</b>:  
            - Stay indoors with medical-grade air purifiers and sealed windows to avoid pollutant exposure.  
            - Monitor for symptoms like dizziness or reduced fetal movement and contact your doctor immediately.  
            - Rest frequently and avoid any stress or exertion.  
            <b>Pro Tip</b>: Use a fetal heart monitor at home if recommended by your doctor to ensure baby’s well-being.
        """
    },
    'Severe': {
        'General': """
            <b>Health</b>: Emergency conditions with significant health risks for all, including severe respiratory and heart issues.  
            <b>Actions</b>: Remain indoors with all windows and doors sealed, avoiding any physical activity.  
            <b>Tips</b>:  
            - Use medical-grade air purifiers to ensure safe indoor air quality.  
            - Seek immediate medical help if you experience breathing difficulties or heart palpitations.  
            - Avoid cooking methods that produce smoke, like frying, to minimize indoor pollutants.  
            <b>Pro Tip</b>: Become an advocate for stricter air quality regulations to prevent severe AQI events!
        """,
        'Asthma': """
            <b>For Asthma Patients</b>: Severe air quality can lead to critical asthma emergencies.  
            <b>Tips</b>:  
            - Stay indoors with sealed windows and run medical-grade air purifiers continuously.  
            - Follow your asthma action plan strictly and keep emergency medications ready.  
            - Seek immediate medical attention for severe symptoms like inability to speak or severe wheezing.  
            <b>Pro Tip</b>: Ensure a family member or caregiver is aware of your condition and can assist in emergencies.
        """,
        'Heart Disease': """
            <b>For Heart Disease Patients</b>: Severe air quality can trigger life-threatening cardiovascular events.  
            <b>Tips</b>:  
            - Remain indoors with air purifiers and avoid any physical or emotional stress.  
            - Monitor for symptoms like chest pain, shortness of breath, or irregular heartbeats; seek emergency care if needed.  
            - Keep all medications accessible and ensure a calm indoor environment.  
            <b>Pro Tip</b>: Have a hospital emergency plan ready, including transport options, for immediate response.
        """,
        'Elderly': """
            <b>For the Elderly</b>: Severe air quality poses extreme risks, potentially leading to critical health events.  
            <b>Tips</b>:  
            - Stay indoors with air purifiers running and avoid any activity, even light movement.  
            - Monitor for severe symptoms like confusion, chest pain, or breathing difficulties; seek emergency care immediately.  
            - Ensure caregivers or family check in regularly.  
            <b>Pro Tip</b>: Keep a medical alert device or phone nearby for quick access to emergency services.
        """,
        'Children': """
            <b>For Children</b>: Severe air quality can cause acute respiratory distress and long-term health impacts.  
            <b>Tips</b>:  
            - Keep children indoors with medical-grade air purifiers in all rooms.  
            - Monitor for signs of respiratory distress, lethargy, or coughing; contact a pediatrician immediately if symptoms appear.  
            - Avoid any physical activities and maintain a calm indoor environment.  
            <b>Pro Tip</b>: Use comforting activities like reading or watching movies to keep children calm and distracted.
        """,
        'Pregnancy': """
            <b>For Pregnant Individuals</b>: Severe air quality can seriously harm maternal and fetal health.  
            <b>Tips</b>:  
            - Stay indoors with medical-grade air purifiers and sealed windows to block all pollutants.  
            - Monitor for symptoms like severe fatigue, dizziness, or reduced fetal movement; seek emergency care if needed.  
            - Rest as much as possible and avoid any stress or exertion.  
            <b>Pro Tip</b>: Keep your obstetrician’s contact information handy and use telehealth for immediate consultations.
        """
    }
}

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

st.title("🌿 Air Quality Index (AQI) Dashboard")
st.markdown("Explore air quality trends across cities with interactive visualizations, predictions, and personalized AQI assistance.")

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
            "🚨 Live AQI Alerts",
            "🌱 AQI Assistant"
        ],
        format_func=lambda x: x[2:]
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
    st.info("Select a view to explore AQI data, predict air quality, or get personalized assistance.")

if page == "📊 City-wise AQI":
    st.header("📊 City-wise AQI Trends")
    col1, col2 = st.columns([3, 1])
    with col1:
        city = st.selectbox("Select a city", all_cities, key="city_select")
    
    city_df = df[df['City'] == city]
    
    if not city_df.empty:
        with st.spinner("Loading AQI trend..."):
            fig, ax = plt.subplots(figsize=(10, 4))
            monthly_aqi = city_df.groupby('Month')['AQI'].mean().reindex(
                ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            )
            ax.plot(monthly_aqi.index, monthly_aqi.values, color='blue')
            ax.set_title(f'Average AQI Trend for {city}')
            ax.set_xlabel('Month')
            ax.set_ylabel('AQI')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        st.subheader("AQI Summary Statistics")
        summary = city_df['AQI'].agg(['mean', 'min', 'max']).to_frame().T
        summary.columns = ['Mean AQI', 'Min AQI', 'Max AQI']
        st.table(summary.round(2))
        
        st.subheader("Pollutant Contribution")
        pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3']
        pollutant_means = city_df[pollutants].mean()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(pollutant_means, labels=pollutants, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
        
        csv = city_df.to_csv(index=False)
        st.download_button(
            label="Download City Data as CSV",
            data=csv,
            file_name=f"{city}_aqi_data.csv",
            mime="text/csv"
        )
    else:
        st.warning(f"No historical AQI data available for {city}. Try checking live AQI in the 'Live AQI Alerts' page.")
    
    st.markdown(f"Trend Analysis for {city}: Visualize AQI fluctuations and pollutant contributions.")

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
        
        if st.button("Predict AQI", key="predict_button"):
            with st.spinner("Predicting..."):
                time.sleep(1)
                input_data = [pm25, pm10, no2, co, o3]
                predicted_aqi = model.predict([input_data])[0]
                aqi_category = get_aqi_category(predicted_aqi)
                st.write(f"Predicted AQI: {predicted_aqi:.2f}")
                st.markdown(f"<span class='{get_aqi_category_class(aqi_category)}'>Air Quality Category: <b>{aqi_category}</b></span>", unsafe_allow_html=True)
                st.balloons()
                st.markdown("<h3>AQI Assistant</h3>", unsafe_allow_html=True)
                st.markdown(f"<div class='chatbot-message'>{aqi_recommendations.get(aqi_category, {}).get('General', 'No recommendations available.')}</div>", unsafe_allow_html=True)
                st.markdown(long_term_consequences, unsafe_allow_html=True)

elif page == "🆚 Compare Cities":
    st.header("🆚 Compare AQI Between Cities")
    
    col1, col2 = st.columns(2)
    with col1:
        city1 = st.selectbox("City 1", all_cities, index=0, key="city1_select")
    with col2:
        city2 = st.selectbox("City 2", all_cities, index=1, key="city2_select")
    
    city1_df = df[df['City'] == city1]
    city2_df = df[df['City'] == city2]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"AQI Trend: {city1}")
        if not city1_df.empty:
            with st.spinner(f"Loading {city1} data..."):
                fig, ax = plt.subplots(figsize=(8, 3.5))
                monthly_aqi = city1_df.groupby('Month')['AQI'].mean().reindex(
                    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                )
                ax.plot(monthly_aqi.index, monthly_aqi.values, color='blue')
                ax.set_title(f'Average AQI Trend for {city1}')
                ax.set_xlabel('Month')
                ax.set_ylabel('AQI')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning(f"No historical AQI data available for {city1}.")
    
    with col2:
        st.subheader(f"AQI Trend: {city2}")
        if not city2_df.empty:
            with st.spinner(f"Loading {city2} data..."):
                fig, ax = plt.subplots(figsize=(8, 3.5))
                monthly_aqi = city2_df.groupby('Month')['AQI'].mean().reindex(
                    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                )
                ax.plot(monthly_aqi.index, monthly_aqi.values, color='blue')
                ax.set_title(f'Average AQI Trend for {city2}')
                ax.set_xlabel('Month')
                ax.set_ylabel('AQI')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning(f"No historical AQI data available for {city2}.")
    
    combined_df = pd.concat([city1_df, city2_df])
    if not combined_df.empty:
        csv = combined_df.to_csv(index=False)
        st.download_button(
            label="Download Comparison Data as CSV",
            data=csv,
            file_name=f"{city1}_vs_{city2}_aqi_data.csv",
            mime="text/csv"
        )

elif page == "🔥 Heatmap":
    st.header("🔥 AQI Heatmap by Month and City")
    with st.spinner("Generating heatmap..."):
        heat_df = df.copy()
        heat_df['Month'] = heat_df['Date'].dt.month_name()
        pivot = heat_df.pivot_table(index='City', columns='Month', values='AQI', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(pivot, cmap="YlOrRd", ax=ax, annot=True, fmt=".1f", cbar_kws={'label': 'AQI'})
        ax.set_title("Average AQI by City and Month")
        plt.tight_layout()
        st.pyplot(fig)
    st.markdown("Insight: Red indicates higher AQI (worse air quality). Compare monthly patterns across cities.")

elif page == "🏆 Top 10 Polluted Cities":
    st.header("🏆 Top 10 Most Polluted Cities")
    with st.spinner("Calculating rankings..."):
        avg_aqi = df.groupby('City')['AQI'].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(avg_aqi.index, avg_aqi.values, color='red')
        ax.set_title("Top 10 Most Polluted Cities")
        ax.set_xlabel('Average AQI')
        ax.set_ylabel('City')
        plt.tight_layout()
        st.pyplot(fig)
    st.markdown("Insight: These cities have the highest average AQI, indicating poorer air quality.")

elif page == "🚨 Live AQI Alerts":
    st.header("🚨 Live AQI Alerts")
    st.markdown("Check real-time AQI for a selected city and receive alerts if air quality is poor.")

    city = st.selectbox("Select a city for live AQI", all_cities, key="live_city_select")
    
    with st.spinner(f"Fetching live AQI for {city}..."):
        live_aqi = get_live_aqi(city.lower())
        if live_aqi is not None:
            aqi_category = get_aqi_category(live_aqi)
            st.write(f"Live AQI for {city}: {live_aqi:.2f}")
            st.markdown(f"<span class='{get_aqi_category_class(aqi_category)}'>Air Quality Category: <b>{aqi_category}</b></span>", unsafe_allow_html=True)
            
            weather = get_weather_data(city.lower())
            if weather:
                st.markdown(f"""
                    <h3>Weather Context for {city}</h3>
                    Temperature: {weather['temperature']}°C<br>
                    Humidity: {weather['humidity']}%<br>
                    Wind Speed: {weather['wind_speed']} m/s<br>
                    <i>High humidity and low wind speed can trap pollutants, worsening AQI.</i>
                """, unsafe_allow_html=True)
            
            if live_aqi > 100:
                st.markdown(f"<div class='alert-high-aqi'>⚠️ High AQI Alert: Take precautions as air quality is {aqi_category.lower()}!</div>", unsafe_allow_html=True)
            
            st.markdown("<h3>AQI Assistant</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='chatbot-message'>{aqi_recommendations.get(aqi_category, {}).get('General', 'No recommendations available.')}</div>", unsafe_allow_html=True)
            
            st.markdown(long_term_consequences, unsafe_allow_html=True)
        else:
            st.error(f"Unable to fetch live AQI data for {city}. This city may not have an active monitoring station. Try another city or check your connection.")

elif page == "🌱 AQI Assistant":
    st.header("🌱 Advanced AQI Assistant")
    st.markdown("Your personal AQI Assistant provides tailored recommendations and estimates the impact of environmental actions like tree planting or car removal.")

    st.subheader("Your Profile")
    col1, col2 = st.columns(2)
    with col1:
        current_aqi = st.number_input("Current AQI", 0.0, 500.0, 100.0, step=1.0)
        health_conditions = st.multiselect(
            "Select Health Conditions",
            ["None", "Asthma", "Heart Disease", "Elderly", "Children", "Pregnancy"],
            default=["None"]
        )
    with col2:
        city = st.selectbox("Select City", all_cities, key="assistant_city_select")
        action = st.selectbox("Select Action to Explore", ["None", "Plant Trees", "Remove Cars"])

    weather = get_weather_data(city.lower())
    if weather:
        st.markdown(f"""
            <h3>Weather Context for {city}</h3>
            Temperature: {weather['temperature']}°C<br>
            Humidity: {weather['humidity']}%<br>
            Wind Speed: {weather['wind_speed']} m/s<br>
            <i>High humidity and low wind speed can trap pollutants, worsening AQI.</i>
        """, unsafe_allow_html=True)

    st.subheader("Personalized Health Recommendations")
    aqi_category = get_aqi_category(current_aqi)
    if "None" in health_conditions:
        recommendation = aqi_recommendations.get(aqi_category, {}).get('General', "No recommendations available.")
    else:
        recommendation = f"<b>General Advice for {aqi_category} AQI</b>:<br>{aqi_recommendations.get(aqi_category, {}).get('General', 'No general recommendations available.')}<br><br>"
        recommendation += f"<b>Personalized Advice for Your Health Conditions ({', '.join(health_conditions)})</b>:<br>"
        for condition in health_conditions:
            condition_recommendation = aqi_recommendations.get(aqi_category, {}).get(condition, f"No specific recommendations for {condition}.")
            recommendation += f"{condition_recommendation}<br><br>"
    st.markdown(f"<div class='chatbot-message'>{recommendation}</div>", unsafe_allow_html=True)
    st.markdown(long_term_consequences, unsafe_allow_html=True)

    if action != "None":
        st.subheader(f"Impact of {action}")
        if action == "Plant Trees":
            num_trees = st.slider("Number of Trees (10,000 - 100,000)", 10000, 100000, 10000, step=1000)
            impact = estimate_tree_impact(num_trees, current_aqi)
            st.markdown(f"""
                <div class='assistant-response'>
                <b>Impact of Planting {num_trees:,} Trees in {city}</b><br>
                - Estimated AQI Reduction: {current_aqi - impact['new_aqi']:.2f} points<br>
                - New AQI: {impact['new_aqi']:.2f} ({impact['category']})<br>
                - PM2.5 Reduction: {impact['pm25_reduction']:.2f} µg/m³<br>
                - CO2 Reduction: {impact['co2_reduction']:.2f} metric tons/year<br>
                <b>Context</b>: Trees absorb PM2.5 and CO2, improving air quality over time. Planting native species like pine or cedar can enhance local ecosystems.
                </div>
            """, unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(['Current AQI', 'New AQI'], [current_aqi, impact['new_aqi']], color=['red', 'green'])
            ax.set_title(f'AQI Before and After Planting {num_trees:,} Trees')
            ax.set_ylabel('AQI')
            plt.tight_layout()
            st.pyplot(fig)

        elif action == "Remove Cars":
            num_cars = st.slider("Number of Cars Removed (1,000 - 10,000)", 1000, 10000, 1000, step=100)
            impact = estimate_car_removal_impact(num_cars, current_aqi)
            st.markdown(f"""
                <div class='assistant-response'>
                <b>Impact of Removing {num_cars:,} Cars in {city}</b><br>
                - Estimated AQI Reduction: {current_aqi - impact['new_aqi']:.2f} points<br>
                - New AQI: {impact['new_aqi']:.2f} ({impact['category']})<br>
                - PM2.5 Reduction: {impact['pm25_reduction']:.2f} µg/m³<br>
                - NO2 Reduction: {impact['no2_reduction']:.2f} µg/m³<br>
                - CO2 Reduction: {impact['co2_reduction']:.2f} metric tons/year<br>
                <b>Context</b>: In your city, vehicular emissions are a major pollution source. Reducing cars promotes cleaner air and reduces traffic-related PM2.5 and NO2.
                </div>
            """, unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(['Current AQI', 'New AQI'], [current_aqi, impact['new_aqi']], color=['red', 'green'])
            ax.set_title(f'AQI Before and After Removing {num_cars:,} Cars')
            ax.set_ylabel('AQI')
            plt.tight_layout()
            st.pyplot(fig)
