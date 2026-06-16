import streamlit as st
import os
import numpy as np 
import joblib 
import pandas as pd

st.set_page_config(page_title="network intrusion detection system", layout="wide")

@st.cache_resource
def load_model():
    return {
    'Logistic Regression': joblib.load("logistic_regression.pkl"),
    'Random Forest': joblib.load("random_forest.pkl"),
    'decision tree': joblib.load("decision_tree.pkl"),
    }, joblib.load("standard_scaler.pkl")


models, scaler = load_model()

st.title("Network Intrusion Detection System")
st.caption("Classify network traffic as normal or attack using machine learning models.")

col1, col2 =st.columns([1,2])

with col1:
    st.subheader("Quick Test Scenarios")
    scenario = st.selectbox("Load a scenario", [
        "Custom Input",
        "Normal Traffic",
        "DoS Attack",
        "Probe Attack",
    ])

    # Auto fill values based on scenario
    defaults = {
        "Custom Input":   [0, 0, 10, 10, 1.0, 0.0, 50, 1.0],
        "Normal Traffic": [1000, 2000, 5, 5, 0.9, 0.1, 50, 0.9],
        "DoS Attack":     [2000000, 0, 512, 512, 1.0, 0.0, 255, 1.0],
        "Probe Attack":   [0, 0, 300, 1, 0.05, 0.95, 255, 0.1],
    }

    vals = defaults[scenario]
    src_bytes              = st.number_input("Source Bytes", 0, 5000000, vals[0])
    dst_bytes              = st.number_input("Destination Bytes", 0, 5000000, vals[1])
    count                  = st.number_input("Count", 0, 512, vals[2])
    srv_count              = st.number_input("Srv Count", 0, 512, vals[3])
    same_srv_rate          = st.number_input("Same Service Rate", 0.0, 1.0, vals[4], step=0.01)
    diff_srv_rate          = st.number_input("Different Service Rate", 0.0, 1.0, vals[5], step=0.01)
    dst_host_srv_count     = st.number_input("Dst Host Service Count", 0, 255, vals[6])
    dst_host_same_srv_rate = st.number_input("Dst Host Same Service Rate", 0.0, 1.0, vals[7], step=0.01)

with col2:
    if st.button("Predict", use_container_width=True):
        feature_cols = joblib.load('feature_columns.pkl')
        feat_dict = {col: 0 for col in feature_cols}

        feat_dict['src_bytes']               = src_bytes
        feat_dict['dst_bytes']               = dst_bytes
        feat_dict['count']                   = count
        feat_dict['srv_count']               = srv_count
        feat_dict['same_srv_rate']           = same_srv_rate
        feat_dict['diff_srv_rate']           = diff_srv_rate
        feat_dict['dst_host_srv_count']      = dst_host_srv_count
        feat_dict['dst_host_same_srv_rate']  = dst_host_same_srv_rate

        feat_input = pd.DataFrame([feat_dict])
        model = models[model_name]

        if model_name == 'Logistic Regression':
            feat_scaled = pd.DataFrame(
                scaler.transform(feat_input),
                columns=feature_cols
            )
            pred  = model.predict(feat_scaled)[0]
            proba = model.predict_proba(feat_scaled)[0]
        else:
            pred  = model.predict(feat_input)[0]
            proba = model.predict_proba(feat_input)[0]

        conf  = round(max(proba) * 100, 1)
        color = "green" if pred == "normal" else "red"
        st.markdown(f"### Prediction: :{color}[{str(pred).upper()}]")
        st.metric("Confidence", f"{conf}%")

        classes = model.classes_
        prob_df = pd.DataFrame({'Class': classes, 'Probability': proba})
        st.bar_chart(prob_df.set_index('Class'))
