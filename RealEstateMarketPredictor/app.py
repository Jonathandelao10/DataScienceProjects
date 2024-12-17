import streamlit as st
import numpy as np
import joblib

scaler = joblib.load("Scaler.pkl")

model = joblib.load("model.pkl")

st.title("Real Estate Price Prediction App")

st.divider() # To make it look prettier

bed = st.number_input("Enter the number of bedrooms", value=2, step=1)
bath = st.number_input("Enter the number of bathrooms", value=1, step=1)
size = st.number_input("Enter the size", value=1000, step=50)

X = [bed, bath, size]

st.divider()

predictButton = st.button("Predict")

st.divider()

if predictButton: 
    st.balloons()
    
    X1 = np.array(X)
    
    X_array = scaler.transform([X1])
    
    prediction = model.predict(X_array)[0]
    
    st.write(f"The prediction is: {prediction:.2f}")
    
else:
    "Please use the button for prediction"

# ['bed', 'bath', 'house_size']
