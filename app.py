import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Network Intrusion Detection System", layout="wide")

@st.cache_resource
def load_model():
    return {
        'Logistic Regression': joblib.load("logistic_regression.pkl"),
        'Random Forest':       joblib.load("random_forest.pkl"),
        'Decision Tree':       joblib.load("decision_tree.pkl"),
    }, joblib.load("standard_scaler.pkl")

models, scaler = load_model()
feature_cols = joblib.load("feature_columns.pkl")

# Real rows from NSL-KDD dataset — all 41 features
SCENARIOS = {
    "Custom Input": None,
    "Normal Traffic": [0.0,1.0,20.0,9.0,491.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,2.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,150.0,25.0,0.17,0.03,0.17,0.0,0.0,0.0,0.05,0.0],
    "DoS Attack":    [0.0,1.0,49.0,5.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,123.0,6.0,1.0,1.0,0.0,0.0,0.05,0.07,0.0,255.0,26.0,0.1,0.05,0.0,0.0,1.0,1.0,0.0,0.0],
    "Probe Attack":  [0.0,0.0,14.0,9.0,18.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,16.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0],
    "R2L Attack":    [0.0,1.0,20.0,9.0,334.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,2.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,2.0,20.0,1.0,0.0,1.0,0.2,0.0,0.0,0.0,0.0],
    "U2R Attack":    [98.0,1.0,60.0,9.0,621.0,8356.0,0.0,0.0,1.0,1.0,0.0,1.0,5.0,1.0,0.0,14.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,255.0,4.0,0.02,0.02,0.0,0.0,0.0,0.0,0.0,0.0],
}

st.title("Network Intrusion Detection System")
st.caption("Classify network traffic as normal or attack using machine learning models.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Select Model")
    model_name = st.selectbox("", list(models.keys()))

    st.subheader("Load a Scenario")
    scenario = st.selectbox("Quick test", list(SCENARIOS.keys()))

    st.subheader("Input Features")

    # If scenario selected, use real values — else default to 0
    vals = SCENARIOS[scenario] if SCENARIOS[scenario] else [0] * 41

    src_bytes              = st.number_input("Source Bytes",                   0, 5000000, int(vals[4]))
    dst_bytes              = st.number_input("Destination Bytes",              0, 5000000, int(vals[5]))
    count                  = st.number_input("Count",                          0, 512,     int(vals[22]))
    srv_count              = st.number_input("Srv Count",                      0, 512,     int(vals[23]))
    same_srv_rate          = st.number_input("Same Service Rate",              0.0, 1.0,   float(vals[28]), step=0.01)
    diff_srv_rate          = st.number_input("Different Service Rate",         0.0, 1.0,   float(vals[29]), step=0.01)
    dst_host_srv_count     = st.number_input("Destination Host Service Count", 0, 255,     int(vals[32]))
    dst_host_same_srv_rate = st.number_input("Destination Host Same Srv Rate", 0.0, 1.0,   float(vals[33]), step=0.01)

with col2:
    if st.button("Predict", use_container_width=True):
        model = models[model_name]

        if SCENARIOS[scenario]:
            # Use full 41-feature real row for preset scenarios
            feat_input = pd.DataFrame([SCENARIOS[scenario]], columns=feature_cols)
        else:
            # Custom input — use 8 features, rest 0
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
