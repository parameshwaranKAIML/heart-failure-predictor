 # 🩺 CardioCare AI: Heart Failure Readmission Predictor

A modern, interactive machine learning web application designed to evaluate 30-day heart failure readmission risk using clinical biomarkers. 
Built as a personal learning project to master end-to-end ML workflows, from model training to cloud deployment.

🔗 **Live Application:**  [View Live App on Streamlit](https://heart-failure-predictor-ytenayz73kpjnxs8szd5dn.streamlit.app/)


## 🌟 Project Overview
Real-world clinical decision support requires more than just binary answers. 
This project implements a **3-tier risk classification system** (Low, Moderate, and High Risk) backed by a Random Forest classifier.
The app features a professional medical dashboard layout with a clean dark sidebar, collapsible input parameters, and live summary metric cards.



## 🧠 Features & Architecture
 **Machine Learning Model:** Uses a serialized *Random Forest Classifier* (`heart_failure_model.pkl`) trained on clinical diagnostic features.
 **3-Tier Risk Evaluation:** Custom probability threshold logic (`app.py`) that categorizes patients into:
  * 🟢 **Low Risk:** Stable biometric indicators.
  * 🟡 **Moderate / Borderline Risk:** Requires careful monitoring and follow-up.
  * 🔴 **High Risk:** High-priority clinical intervention recommended.
* **Interactive UI/UX:** Built with **Streamlit**, featuring custom CSS styling, wide dashboard layout, and dynamic metric delta indicators.



## 📋 Input Clinical Parameters
The app collects and analyzes the following patient metrics:
* **Demographics & Habits:** Age, Biological Sex, Smoking Status.
* **Clinical Biomarkers:** Ejection Fraction (%), Serum Creatinine (mg/dL), Serum Sodium (mEq/L), Platelet Count.
* **Medical History & Timeline:** Follow-up Window (days), High Blood Pressure, Diabetes, Anaemia.



## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn, Joblib, Pandas, NumPy
* **Web Framework:** Streamlit
* **Version Control & Hosting:** Git, GitHub, Streamlit Community Cloud
