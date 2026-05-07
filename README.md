# 🏭 Industrial Predictive Maintenance Dashboard

## 📌 Project Overview
This project is an Industrial Engineering predictive maintenance system that uses machine learning to detect machine failure risks and monitor production health through an interactive dashboard.

The system uses **Logistic Regression** to predict machine failure based on sensor and operational data.

---

## 📌 Executive Summary

This dashboard helps industrial plant floor managers monitor machine health in real time and predict potential failures before they occur. By using a Logistic Regression-based predictive model, the system provides early warnings for high-risk machines through clear visual indicators, control charts, and anomaly alerts.

The dashboard is designed to reduce unplanned downtime by enabling proactive maintenance decisions. It also helps maximize production yield by ensuring machines operate within safe performance thresholds. With intuitive visualizations and automated risk scoring, managers can quickly identify issues and prioritize maintenance actions efficiently.

---

## 📊 Features
- Real-time KPI monitoring
- Machine failure prediction
- Risk probability analysis
- Control chart (SPC-style monitoring)
- Feature importance / coefficient analysis
- High-risk machine detection
- Interactive dashboard built with Streamlit
- CSV export functionality

---

## 📂 Dataset
- Source: Kaggle - AI4I 2020 Predictive Maintenance Dataset  
- Target Variable: Machine Failure  
  - `0` = No Failure  
  - `1` = Failure  

---

## 🧠 Machine Learning Model
- Algorithm: Logistic Regression  
- Type: Binary Classification  
- Output: Probability of machine failure  
- Evaluation Metrics: Accuracy, Precision, Recall  

---

## 🚀 How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Run the dashboard
```bash
python -m streamlit run dashboard/app.py
```
### 3. Open in browser
Go to:
http://localhost:8501

---

### Authors
- Algozo, Jairos Joash
- Almalvez, Aira Shane
- Elamparo, Aaron Rafael
- Magnate, Janelle
- Monreal, Angel
