# Network Intrusion Detection System

Classifies network traffic as **Normal** or one of 4 attack types — DoS, Probe, R2L, U2R — using three machine learning models trained on the NSL-KDD benchmark dataset.

**Live Demo:** http://localhost:8501/
**Dataset:** [NSL-KDD on Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd)

---

## Problem Statement

Network intrusion detection is a critical cybersecurity challenge. Traditional rule-based systems fail to detect novel attack patterns. This project applies supervised machine learning to automatically classify network traffic into 5 categories — enabling faster, automated threat detection without manual rule updates.

---

## Model Results

| Model               | Accuracy | F1 Score (Macro) | Precision | Recall |
|---------------------|----------|------------------|-----------|--------|
| Logistic Regression | 94%      | 0.74             | 0.81      | 0.70   |
| Decision Tree       | 94%      | 0.73             | 0.70      | 0.94   |
| Random Forest       | 99%      | 0.94             | 0.98      | 0.92   |

**Random Forest achieved the highest performance** across all metrics. See analysis below.

---

## Why Random Forest Outperforms the Other Models

Logistic Regression performs worse because it can only learn straight-line boundaries, while network attacks have complex patterns.
Decision Tree can learn these patterns but often overfits the training data.
Random Forest combines many decision trees, making predictions more accurate and stable, while also handling rare attack types better.

---

## Attack Categories

| Category |Examples |
|----------|---------|
| Normal   | — |
| DoS      | neptune, smurf, teardrop |
| Probe    | nmap, ipsweep, portsweep |
| R2L      | guess_passwd, ftp_write |
| U2R      | buffer_overflow, rootkit |

---

## EDA Highlights

### Class Distribution
<img width="868" height="871" alt="image" src="https://github.com/user-attachments/assets/f2742b4b-f908-40b3-82f4-ee8d07250204" />

The dataset is heavily imbalanced — Normal and DoS dominate while U2R and R2L are rare. This was handled using `class_weight='balanced'` in all three models.

### Feature Importance (Random Forest)
<img width="1700" height="1582" alt="image" src="https://github.com/user-attachments/assets/e3df1fb1-a3c3-454d-8918-e237e2bb180e" />

Top features driving predictions: `src_bytes`, `dst_host_serror_rate`, `dst_bytes`, `dst_host_same_srv_rate`, `srv_count`, 'count'.

### Correlation Heatmap
<img width="1296" height="989" alt="image" src="https://github.com/user-attachments/assets/6486a507-16ff-4aa7-a70b-00f34bbd1a6d" />


---

## Project Structure

```
nids_project/
├── data/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
├── notebook/
│   └── NSL_KDD Project.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Scikit-learn | Model training and evaluation |
| Pandas / NumPy | Data preprocessing and wrangling |
| Matplotlib / Seaborn | EDA visualisation |
| Streamlit | Web app deployment |
| Joblib | Model serialisation |
| NSL-KDD | Benchmark dataset |
| VS-Code| Code Editor |

---

`

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
imbalanced-learn
joblib
```

---

## Key Learnings

- **Data leakage:** The `difficulty` column was dropped because it was assigned by researchers after labelling — keeping it would inflate accuracy artificially by leaking classification difficulty into the features.
- **Class imbalance:** U2R and R2L attacks represent <1% of records. Using `class_weight='balanced'` was more appropriate than oversampling for this dataset size.
- **Model comparison:** Training all three models and comparing results provides more insight than picking one algorithm upfront.

---

## About

Built by Jay Parmar as part of a machine learning portfolio project.  
Background in System Administration (Linux, networking, VPN) informed the domain understanding applied in this project.

[LinkedIn](https://www.linkedin.com/in/jay-parmar-ml) · [GitHub](https://github.com/jayparmar1)
