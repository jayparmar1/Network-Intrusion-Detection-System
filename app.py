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
    'Random Forest': joblib.load("random_forest.joblib"),
    'decision tree': joblib.load("decision_tree.pkl"),
    }, joblib.load("standard_scaler.pkl")


models, scaler = load_model()

st.title("Network Intrusion Detection System")
st.caption("Classify network traffic as normal or attack using machine learning models.")

col1, cols2 =st.columns([1,2])

with col1:
    st.subheader("select model")
    model_name=st.selectbox("", list(models.keys()))   

    st.subheader("Input Features")
    src_bytes = st.number_input("Source Bytes",0,5000000,0)
    dst_bytes = st.number_input("Destination Bytes",0,5000000,0)
    count = st.number_input("Count",0,512,10)
    srv_count = st.number_input("Srv Count",0,512,10)
    same_srv_rate = st.number_input("Same Service Rate",0.0,1.0,1.0,step=0.01)
    diff_srv_rate = st.number_input("Different Service Rate",0.0,1.0,0.0,step=0.01)
    dst_host_srv_count = st.number_input("Destination Host service Count",0,255,50)
    dst_host_same_srv_rate = st.number_input("Destination Host same service rate",0.0,1.0,1.0,step=0.01)


with cols2:
    if st.button("Predict",use_container_width=True):
        feat=np.zeros(41)
        feat[4]=src_bytes
        feat[5]=dst_bytes
        feat[22]=count
        feat[23]=srv_count
        feat[28]=same_srv_rate
        feat[29]=diff_srv_rate
        feat[32]=dst_host_srv_count
        feat[33]=dst_host_same_srv_rate


        model= models[model_name]
        if model_name == 'Logistic Regression':
            feat_input=scaler.transform([feat])
        else:
            feat_input=[feat]


        prediction=model.predict(feat_input)[0]
        probability=model.predict_proba(feat_input)[0]
        conf=round(max(probability)*100,1)

        color = "green" if prediction == 0 else "red"
        st.markdown(f'### Prediction :{color}[{prediction.upper()}] ')
        st.metric('confidence',f'{conf}%')


        classes=model.classes_
        prob_df=pd.DataFrame({'class':classes,'probability':probability})
        st.bar_chart(prob_df.set_index('class'))