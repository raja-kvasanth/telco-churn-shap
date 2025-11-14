# Interpretable Machine Learning for Customer Churn Prediction  
### Using XGBoost + SHAP on the Telco Customer Churn Dataset

This project builds and interprets a customer churn prediction model using  
the **WA_Fn-UseC_-Telco-Customer-Churn.csv** dataset.  
The model is trained with **XGBoost** inside a complete scikit-learn Pipeline  
and interpreted using **SHAP (Shapley Additive Explanations)** to produce  
both global and local explanations.

---

## 📌 Project Objectives

1. Build a high-quality churn prediction model using advanced ML techniques  
   (XGBoost with hyperparameter tuning).
2. Apply **SHAP interpretability** to extract:
   - Global feature importance  
   - Local explanations for specific customers (waterfall + force plots)
3. Generate business insights and actionable retention recommendations.
4. Provide a transparent, reproducible pipeline for model training, evaluation,  
   and SHAP analysis.

---

## 📂 Repository Structure

telco-churn-shap/
├─ churn_shap_pipeline.ipynb # Colab notebook with full code
├─ churn_shap_pipeline.py # Optional: Python script version
├─ selected_customers.csv # 3 representative customers (low/medium/high risk)
├─ report.json # Test metrics, selected customers info
├─ gitingest_template.md # Auto-generated submission template
├─ requirements.txt # Dependencies
├─ README.md # This file
└─ outputs/
├─ shap_summary_bar.png
├─ shap_summary_beeswarm.png
├─ local_shap_waterfall_idx_971.png
├─ local_shap_waterfall_idx_2034.png
├─ local_shap_waterfall_idx_3676.png
├─ local_shap_force_idx_971.html
├─ local_shap_force_idx_2034.html
├─ local_shap_force_idx_3676.html
└─ xgb_pipeline.joblib


---

## 🧠 Model Overview

- **Model:** XGBoost Classifier  
- **Preprocessing:**  
  - StandardScaler (numeric features)  
  - OneHotEncoder (categorical features)  
- **Hyperparameter Tuning:** RandomizedSearchCV (AUC scoring)  
- **Evaluation Metrics:** Accuracy, AUC, F1  
- **Interpretability:** SHAP TreeExplainer

The entire training process is encapsulated in a scikit-learn Pipeline  
for reproducibility and easy deployment.

---

## 📊 SHAP Interpretability Outputs

The project generates:

### **Global Explanations**
- `shap_summary_bar.png`
- `shap_summary_beeswarm.png`

These plots reveal the overall feature ranking and how feature values  
impact churn predictions across the dataset.

### **Local Explanations (Individual Customers)**
For three selected customers (low, marginal, high risk):

- SHAP **waterfall** plots → PNG  
- SHAP **force** plots → HTML  

These files explain *why* the model predicted each customer's churn probability.

---

## 📁 Dataset

The dataset used is the standard Telco Customer Churn dataset:  
`WA_Fn-UseC_-Telco-Customer-Churn.csv`

It includes:
- Customer demographics  
- Subscription information  
- Internet & phone service details  
- Billing information  
- Churn label (`Churn`)  

The pipeline is designed to automatically preprocess categorical and numeric  
features using a ColumnTransformer.

---

## ▶️ How to Run the Project

### **1. Install dependencies**


### **2. Run the notebook**
Open **churn_shap_pipeline.ipynb** in Google Colab or locally and run all cells.

### **3. Outputs generated**
- Trained model (`xgb_pipeline.joblib`)
- SHAP global plots
- SHAP local plots
- Evaluation metrics (AUC, Accuracy, F1)
- Selected customers CSV

---

## 📝 Using GitIngest for Submission

To generate a submission markdown with all code:

1. Push this repository to GitHub  
2. Go to: https://gitingest.com  
3. Paste your repository URL (public repo recommended)  
4. Click **Generate Markdown**  
5. Copy and paste the markdown into your project submission box

---

## 💡 Business Insights (Summary)

Key insights from SHAP:

- **Month-to-month contracts** majorly increase churn risk  
- **Low tenure** (new customers) have the highest churn sensitivity  
- Missing services like **OnlineSecurity** or **TechSupport** increase risk  
- **Electronic check** payment method is associated with higher churn  
- Fiber optic users show elevated churn likelihood compared to DSL

---

## 🏁 Executive Recommendations

1. Promote **contract upgrades** with discounts for month-to-month customers  
2. Implement a **“First 90 Days Retention Program”** for very new subscribers  
3. Incentivize **auto-pay or credit card billing** to replace electronic check  

Full reasoning for these recommendations is in the analytical report.

---

## 📬 Contact

If you want to reproduce or extend the model, feel free to open an issue in the repository.

