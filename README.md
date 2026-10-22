
```markdown
# 📊 Telco Customer Churn Prediction Pipeline & Web App

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning project designed to predict customer churn for a telecommunications company. The project addresses data imbalance, handles data leakages via scikit-learn Pipelines, incorporates feature engineering, and compares 6 ML models before deploying the best model through an interactive Streamlit UI.

---

## 📌 Problem Statement & Dataset Overview

Customer churn occurs when customers stop doing business with a company. Identifying churn risks early allows retention teams to intervene proactively.

* **Dataset Size:** 7,043 customers
* **Class Imbalance:**
  * **Retained (Churn = No):** 73.5% (5,174 customers)
  * **Churned (Churn = Yes):** 26.5% (1,869 customers)
* **Naive Baseline Accuracy:** 73.5% (Predicting all customers as "No Churn").

---

## 🛠️ Data Processing & Feature Engineering (`dataprocessing.py`)

Key data processing steps applied to clean the dataset and prevent leakage/multicollinearity:

1. **Logical Imputation for Missing Values:**
   * `TotalCharges` contained 11 blank values corresponding exclusively to new customers (`tenure == 0`).
   * Instead of imputing with the median (~$1,397), values were logically filled with `0.0`.
2. **Categorical Consolidation:**
   * Consolidated "No internet service" and "No phone service" into "No" across 6 sub-service columns to eliminate redundant One-Hot features and prevent **Multicollinearity** in Logistic Regression / Naive Bayes / KNN.
3. **Binary Encoding:**
   * Utilized `drop="if_binary"` in `OneHotEncoder`, reducing encoded feature count from 46 down to 27 clean columns.
4. **Feature Engineering Pipeline (`FunctionTransformer`):**
   * **`TotalServices`:** Total number of active subscribed services per customer.
   * **`AvgMonthlyCharges`:** Historical average spend (`TotalCharges / tenure`).
   * **`ChargeDifference`:** Current bill vs. historical average (`MonthlyCharges - AvgMonthlyCharges`). A large positive value signals potential price-hike churn.

---

## 📈 Model Comparison & Benchmark Results

All models were evaluated using **5-Fold Stratified Cross-Validation** to ensure metric stability.

| Model | Accuracy | Precision | Recall | F1-Score | CV_F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **🏆 Random Forest (Tuned)** | **75.2%** | **0.528** | **0.807** | **0.638** | **0.632** | **0.845** |
| Decision Tree | 70.8% | 0.468 | 0.845 | 0.602 | 0.598 | 0.825 |
| Logistic Regression | 74.8% | 0.518 | 0.801 | 0.629 | 0.625 | 0.844 |
| K-Nearest Neighbors (KNN) | 78.8% | 0.606 | 0.572 | 0.588 | 0.581 | 0.832 |
| Support Vector Machine (SVM) | 74.1% | 0.509 | 0.792 | 0.620 | 0.615 | 0.829 |
| Naive Bayes | 73.8% | 0.505 | 0.789 | 0.616 | 0.610 | 0.824 |

---

## 🎯 Why Random Forest Won (Solving Leaf Overfitting)

Initially, Random Forest performed poorly (F1 = 0.589, Recall = 0.631) despite using `class_weight="balanced"`. 

* **Root Cause:** Without depth limits (`max_depth=None`, `min_samples_leaf=1`), individual trees grew until pure single-sample leaf nodes were formed. Pure single-sample leaves invalidate `class_weight="balanced"` during training, leading to severe overfitting toward the majority class.
* **The Fix:** Constrained tree growth using `max_depth=8` and `min_samples_leaf=10` with 300 estimators:
  ```python
  RandomForestClassifier(
      n_estimators=300,
      max_depth=8,
      min_samples_leaf=10,
      class_weight="balanced",
      random_state=42,
      n_jobs=-1
  )

```

* **Result:** Forced multi-sample leaves, allowing `class_weight="balanced"` to function properly. **Recall jumped from 63.1% to 80.7%**, and **F1-Score reached #1 at 0.638**.

---

## 📁 Repository Structure

```text
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
├── src/
│   ├── dataprocessing.py                     # Data cleaning & feature engineering pipeline
│   └── train.py                               # Model training & cross-validation script
├── app.py                                     # Streamlit web interface
├── churn_model.joblib                         # Trained Scikit-Learn pipeline
├── requirements.txt                           # Project dependencies
└── README.md                                  # Project documentation

```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone [https://github.com/Youssef-Mohamed0/Telco-Customer-Churn-Prediction.git](https://github.com/Youssef-Mohamed0/Telco-Customer-Churn-Prediction.git)
cd Telco-Customer-Churn-Prediction

```

### 2. Set Up Virtual Environment & Install Dependencies

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

```

### 3. Train Models

```bash
python src/train.py

```

### 4. Run Streamlit Web Application

```bash
streamlit run app.py

```

---

## 💡 Key Evaluation Metrics Explained

* **Recall over Accuracy:** In telecommunications churn prediction, a **False Negative** (a churner predicted as staying) results in direct revenue loss. Minimizing False Negatives (maximizing **Recall**) is the primary objective.
* **F1-Score:** Used as the final optimization metric to maintain a balance between identifying churners (Recall) and avoiding unnecessary promotional spending on loyal customers (Precision).
* **ROC-AUC (0.845):** Confirms that the model has an 84.5% probability of ranking a random churned customer higher than a random non-churned customer across all decision thresholds.

---

## 👤 Author

* **Youssef Mohamed** - [GitHub](https://www.google.com/search?q=https://github.com/Youssef-Mohamed0) | [LinkedIn](https://www.google.com/search?q=https://www.linkedin.com/in/youssefmohamedegy)

```

```
