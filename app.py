import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page setup
st.set_page_config(page_title="💰 Gold Price Predictor", layout="centered")
st.title("💰 Gold Price Prediction Web App")
st.write("Upload your CSV file or enter values manually to predict the gold price using a trained Random Forest model.")

# Load Random Forest model
try:
    model = joblib.load(r"ML/Gold_Price_Prediction/random_forest.pkl")
except FileNotFoundError:
    st.error(
        "⚠️ Random Forest model not found! Please ensure 'random_forest.pkl' is in the ML folder.")
    st.stop()

# Upload section
st.subheader("📤 Upload a CSV File (Optional)")
uploaded_file = st.file_uploader(
    "Upload a CSV with columns: SPX, USO, SLV, EUR/USD", type=["csv"])

# Load data (uploaded or default)
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Uploaded Data Preview:")
else:
    st.info("No CSV uploaded. Using default sample data.")
    try:
        df = pd.read_csv(r"ML/Gold_Price_Prediction/gold_price.csv")
    except:
        st.error("⚠️ gold_price_data.csv not found.")
        st.stop()

st.write(df.head())

# Predict from file data
try:
    feature_cols = ['SPX', 'USO', 'SLV', 'EUR/USD']
    X_input = df[feature_cols]
    predictions = model.predict(X_input)
    df['Predicted Gold Price'] = predictions

    st.subheader("📈 Predictions from File")
    st.write(df[['Predicted Gold Price']].head())
    st.line_chart(df[['Predicted Gold Price']])

    # Download prediction CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predictions as CSV", data=csv,
                       file_name='predicted_gold_prices.csv', mime='text/csv')

except Exception as e:
    st.error(f"⚠️ Prediction error: {e}")
    st.info("Ensure your CSV has: SPX, USO, SLV, EUR/USD")

# Manual input
st.markdown("---")
st.subheader("🔢 Or Enter Values Manually")

spx = st.number_input("SPX (S&P 500 Index)", value=1400.0)
uso = st.number_input("USO (Oil Price)", value=12.0)
slv = st.number_input("SLV (Silver Price)", value=16.0)
eurusd = st.number_input("EUR/USD Exchange Rate", value=1.2)

if st.button("🎯 Predict Manually"):
    input_data = pd.DataFrame([[spx, uso, slv, eurusd]], columns=feature_cols)
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Gold Price: {prediction:.2f}")
    except Exception as e:
        st.error(f"⚠️ Manual prediction error: {e}")

# Sample CSV download
try:
    with open(r"ML/Gold_Price_Prediction/gold_price.csv", "rb") as file:
        st.download_button("📥 Download Sample CSV", file,
                           file_name="sample_gold_data.csv", mime="text/csv")
except:
    pass
