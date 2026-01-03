import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_lottie import st_lottie
import json
import google.generativeai as genai


# ================= SESSION STATE INITIALIZATION =================
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# ================= GEMINI KEY =================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")


# ================= TITLE =================
st.markdown("""
<h2 style='text-align:center;color:white;text-shadow:0px 0px 10px #00e5ff;'>
🏠 California House Price Prediction
</h2>
""", unsafe_allow_html=True)


# ================= BUTTON STYLE =================
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #3b82f6;
    color:white;
    border-radius: 12px;
    padding: 10px 24px;
    border: 1px solid #1f51ff;
    font-size: 16px;
    font-weight: 600;
    transition: 0.3s;
}
div.stButton > button:first-child:hover {
    background-color: #1f51ff;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)


# ================= LOAD LOTTIE =================
def load_lottie(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return None

animation = load_lottie("Houses in amsterdreams colours.json")
st.subheader("🏡 Buy Your Dream House")
if animation:
    st_lottie(animation, height=300, key="house")


# ================= LOAD MODEL =================
price_model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")


# ================= USER INPUT =================
st.markdown("<h3><b>Enter the house details below and click Predict.</b></h3>", unsafe_allow_html=True)

longitude = st.number_input("Longitude", value=-120.0)
latitude = st.number_input("Latitude", value=35.0)
age = st.number_input("Housing Median Age", value=20.0)
total_rooms = st.number_input("Total Rooms (in Block)", value=1000.0)
total_bedrooms = st.number_input("Total Bedrooms (in Block)", value=200.0)
population = st.number_input("Population (in Block)", value=800.0)
households = st.number_input("Households (in Block)", value=300.0)
median_income = st.number_input("Median Income (in Block)", value=4.0)

ocean = st.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
)

INLAND = 1 if ocean == "INLAND" else 0
ISLAND = 1 if ocean == "ISLAND" else 0
NEAR_BAY = 1 if ocean == "NEAR BAY" else 0
NEAR_OCEAN = 1 if ocean == "NEAR OCEAN" else 0

features = np.array([[longitude, latitude, age,
                      total_rooms, total_bedrooms,
                      population, households, median_income,
                      INLAND, ISLAND, NEAR_BAY, NEAR_OCEAN]])


# ================= PREDICTION =================
if st.button("🔮 Predict Price"):
    with st.spinner("🤖 Predicting..."):
        time.sleep(1)

    scaled = scaler.transform(features)
    prediction = price_model.predict(scaled)[0]

    st.success(f"🏡 Estimated House Price:  ${prediction:,.2f}")
    st.balloons()


# ================= LOAD DATA =================
df = pd.read_csv("housing.csv")


# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("### 🌓 Theme")
    theme = st.toggle("Enable Dark Mode", value=True)

    if theme:
        st.markdown("<style>.stApp{background-color:#0e1117;}</style>", unsafe_allow_html=True)
    else:
        st.markdown("<style>.stApp{background-color:white;}</style>", unsafe_allow_html=True)

    st.markdown("### 📂 Navigation")

    page = st.radio(
        "Navigation",
        [
            "🏡 Home",
            "📊 Dataset Preview",
            "💰 Price Distribution",
            "🔥 Correlation Heatmap",
            "👨‍👩‍👧 Population vs Price",
            "🗺 Interactive Map",
        ],
    )


# ================= PAGE DISPLAY =================
if page == "📊 Dataset Preview":
    st.title("📊 Dataset Preview")
    st.dataframe(df.head())

elif page == "💰 Price Distribution":
    st.title("💰 Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["median_house_value"], kde=True, ax=ax)
    st.pyplot(fig)

elif page == "🔥 Correlation Heatmap":
    st.title("🔥 Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif page == "👨‍👩‍👧 Population vs Price":
    st.title("👨‍👩‍👧 Population vs Price")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df["population"], y=df["median_house_value"], ax=ax)
    st.pyplot(fig)

elif page == "🗺 Interactive Map":
    st.title("🗺 Interactive Map")
    st.map(df.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])


# ========= FLOATING ROUND CHAT BUTTON =========
chat_button_css = """
<style>
.chat-button {
  position: fixed;
  bottom: 25px;
  right: 25px;
  background-color: #3b82f6;
  color: white;
  border-radius: 40px;
  width: 120px;
  height: 65px;
  display: flex;
  justify-content: center;
  align-items: center;
  font-size: 28px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.25);
  cursor: pointer;
  z-index: 9999;
  transition: 0.3s ease-in-out;
}
.chat-button:hover{
  transform: scale(1.1);
}
</style>
"""

st.markdown(chat_button_css, unsafe_allow_html=True)

# Safe toggle
if st.button("💬", key="chat_float_btn"):
    st.session_state.chat_open = not st.session_state.chat_open


# ================= GEMINI CHATBOT =================
if st.session_state.chat_open:

    st.markdown("## 💬 HouseBot — Real Estate Assistant")

    for role, text in st.session_state.chat_messages:
        st.write(f"**{'🧑 You' if role=='user' else '🤖 Bot'}:** {text}")

    user_input = st.chat_input("Ask something...")

    if user_input:

        st.session_state.chat_messages.append(("user", user_input))

        chat_text = ""
        for r, m in st.session_state.chat_messages:
            chat_text += f"{r}: {m}\n"

        response = gemini_model.generate_content(
            f"""
            You are HouseBot, a friendly real-estate chatbot.
            Chat history:
            {chat_text}

            User message:
            {user_input}
            """
        )

        reply = response.text
        st.session_state.chat_messages.append(("assistant", reply))

        st.rerun()


# ================= FOOTER =================
st.markdown(
    "<p style='text-align:center;'>Made with love 💜 <b>Ayan</b></p>",
    unsafe_allow_html=True
)
