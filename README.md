# Network Intrusion Detection System

Classifies network traffic as **Normal** or one of 4 attack types — DoS, Probe, R2L, U2R — using three machine learning models trained on the NSL-KDD benchmark dataset.

**Live Demo:** [your-app.streamlit.app](https://your-app.streamlit.app)  
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

Logistic Regression assumes a linear decision boundary between classes. Network intrusion data is highly non-linear — for example, DoS attacks are defined by extremely high `src_bytes` and `count` values, while Probe attacks show high `diff_srv_rate` with near-zero bytes. These relationships cannot be captured by a linear model, which explains its lower F1 score.

Decision Tree improves on this by learning non-linear splits, but a single tree tends to overfit — it memorises the training data too closely and generalises poorly to unseen traffic patterns.

Random Forest solves both problems. By training 100 decision trees on different random subsets of the data and averaging their predictions, it reduces variance without losing the non-linear expressiveness of individual trees. The `class_weight='balanced'` parameter ensures rare attack types like U2R and R2L — which make up less than 1% of the dataset — still receive adequate attention during training.

---

## Attack Categories

| Category | Description | Examples |
|----------|-------------|---------|
| Normal   | Legitimate network traffic | — |
| DoS      | Flood server to cause crash or unavailability | neptune, smurf, teardrop |
| Probe    | Scan network to find vulnerabilities | nmap, ipsweep, portsweep |
| R2L      | Gain unauthorised local access from remote machine | guess_passwd, ftp_write |
| U2R      | Escalate privileges from user to root/admin | buffer_overflow, rootkit |

---

## EDA Highlights

### Class Distribution
<img width="868" height="862" alt="image" src="https://github.com/user-attachments/assets/e95746cd-216f-45bc-b262-c35891675dbd" />

The dataset is heavily imbalanced — Normal and DoS dominate while U2R and R2L are rare. This was handled using `class_weight='balanced'` in all three models.

### Feature Importance (Random Forest)
<img width="1700" height="1582" alt="image" src="https://github.com/user-attachments/assets/e3df1fb1-a3c3-454d-8918-e237e2bb180e" />

Top features driving predictions: `src_bytes`, `count`, `dst_bytes`, `dst_host_same_srv_rate`, `srv_count`. These align with known attack signatures — DoS attacks generate abnormally high `src_bytes` and `count`, while Probe attacks show distinct `diff_srv_rate` patterns.

### Correlation Heatmap
<img width="1296" height="989" alt="image" src="https://github.com/user-attachments/assets/6486a507-16ff-4aa7-a70b-00f34bbd1a6d" />

Several features showed high correlation (>0.9), particularly among `dst_host_*` rate features. These were retained for tree-based models which are unaffected by multicollinearity, and noted as a candidate for further optimisation on Logistic Regression.

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
├── notebooks/
│   ├── 01_explore.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modelling.ipynb
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

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/nids-project.git
cd nids-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
# Download KDDTrain+.txt and KDDTest+.txt from Kaggle
# Place them in the data/ folder

# 4. Run the notebooks in order
# notebooks/01_explore.ipynb
# notebooks/02_preprocessing.ipynb
# notebooks/03_modelling.ipynb
# This will generate all .pkl files in models/

# 5. Launch the app
streamlit run app.py
```

---

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
- **Feature selection:** Streamlit app inputs were chosen based on Random Forest feature importance scores, not arbitrarily — ensuring the most predictive features are exposed to the user.
- **Model comparison:** Training all three models and comparing results provides more insight than picking one algorithm upfront.

---

## About

Built by Jay Parmar as part of a machine learning portfolio project.  
Background in System Administration (Linux, networking, VPN) informed the domain understanding applied in this project.

[LinkedIn](https://www.linkedin.com/in/jay-parmar-ml) · [GitHub](https://github.com/jayparmar1)
