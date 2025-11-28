
import streamlit as st
import joblib
import pandas as pd

st.title("Construction Time & Cost Prediction")

pre = joblib.load("preprocessor.pkl")
model = joblib.load("lgbm.pkl")

inputs = {}
inputs["land_size"] = st.number_input("Land Size(in -sq.ft)", 0)
inputs["materials_cost"] = st.number_input("Materials Cost (in rs)", 0)
inputs["num_labours"] = st.number_input("Number of Labours", 0)
inputs["labour_efficiency"] = st.slider("Labour Efficiency (avg)", 0.0, 1.0)
inputs["weather_index"] = st.slider("Weather Index", 0.0, 1.0)
inputs["material_shortage_risk"] = st.slider("Material Shortage Risk", 0.0, 1.0)
inputs["demand_supply"] = st.slider("Demand Supply", 0.0, 1.0)
inputs["terrain"] = st.selectbox("Terrain", ["plain","hilly","mountain"])
inputs["project_type"] = st.selectbox("Project Type", ["residential","commercial","industrial"])

if st.button("Predict"):
    df = pd.DataFrame([inputs])
    x = pre.transform(df)
    pred = model.predict(x)[0]
    st.write("Predicted Time (in -days):", pred[0])
    st.write("Predicted Cost (in rs):", pred[1]*80)
