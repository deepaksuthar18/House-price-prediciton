import pickle

import numpy as np
import streamlit as st

# Load the trained best model
with open("best_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the saved scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title("🏡 House Price Prediction App")
st.write("Enter the house details below to predict the price:")

# Feature input fields
area = st.number_input("🏠 Area (in sq. ft):", min_value=500, step=100)
bedrooms = st.number_input("🛏 Bedrooms:", min_value=1, step=1)
bathrooms = st.number_input("🚿 Bathrooms:", min_value=1, step=1)
stories = st.number_input("🏢 Stories:", min_value=1, step=1)
mainroad = st.selectbox("🛣 Main Road Access?", ["Yes", "No"])
guestroom = st.selectbox("🛋 Guest Room Available?", ["Yes", "No"])
basement = st.selectbox("🏠 Basement Available?", ["Yes", "No"])
hotwaterheating = st.selectbox("🔥 Hot Water Heating?", ["Yes", "No"])
airconditioning = st.selectbox("❄ Air Conditioning?", ["Yes", "No"])
parking = st.number_input("🚗 Parking Spaces:", min_value=0, step=1)
prefarea = st.selectbox("🌳 Preferred Area?", ["Yes", "No"])
furnishingstatus = st.selectbox("🛋 Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])

# Convert categorical inputs to numerical (assuming Yes=1, No=0)
mainroad = 1 if mainroad == "Yes" else 0
guestroom = 1 if guestroom == "Yes" else 0
basement = 1 if basement == "Yes" else 0
hotwaterheating = 1 if hotwaterheating == "Yes" else 0
airconditioning = 1 if airconditioning == "Yes" else 0
prefarea = 1 if prefarea == "Yes" else 0

# Convert furnishing status to numerical
furnishing_map = {"Furnished": 2, "Semi-Furnished": 1, "Unfurnished": 0}
furnishingstatus = furnishing_map[furnishingstatus]

# Prepare input for prediction
input_data = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom,
                        basement, hotwaterheating, airconditioning, parking,
                        prefarea, furnishingstatus]])

# Scale input data
input_scaled = scaler.transform(input_data)

# Predict button
if st.button("🏡 Predict House Price"):
    predicted_price = model.predict(input_scaled)[0]
    st.success(f"🏠 Estimated House Price: ${predicted_price:,.2f}")
