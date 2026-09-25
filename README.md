# 🛒 E-Commerce Sales Analytics & Machine Learning Intelligence Suite

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask 3.0](https://img.shields.io/badge/Flask-3.0.3-green.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-orange.svg)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.0-blue.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Data Science, Data Analytics, and Machine Learning project built on Superstore E-Commerce transactional data (~10,000 order records across 793 customers). Features SQL data modeling, exploratory analysis, 3 production-grade ML models, and a real-time Flask web application dashboard.

---

## 🌟 Key Features

### 1. 📈 Time-Series Sales & Revenue Forecasting (Regression)
* Aggregates order streams into weekly continuous series.
* Engineers temporal features (*Year, Month, Quarter, Week of Year*), lag metrics (*1 to 12 weeks*), and rolling statistical windows (*4 to 12 weeks*).
* Model Leaderboard: **Gradient Boosting Regressor** ($R^2 = 0.4319$, MAE = $4,505.86), Linear Regression, Random Forest, XGBoost Regressor.

### 2. 👥 RFM Customer Segmentation (Unsupervised K-Means)
* Calculates **Recency**, **Frequency**, and **Monetary** (RFM) metrics for 793 customers.
* Optimized via Silhouette Analysis into 3 distinct business personas:
  * **Champions / VIPs**: 269 customers ($5,140.99 avg spend, 8.8 orders).
  * **Loyal Customers**: 413 customers ($1,773.81 avg spend).
  * **Lapsed / At-Risk**: 111 customers (531 days inactive).

### 3. ⚠️ Order Profitability & Loss Classification (Supervised Binary ML)
* Predicts whether an incoming order will result in a financial loss (`Profit < 0`) before shipping.
* **XGBoost Classifier**: **95.05% Accuracy**, **93.10% Precision**, **0.9883 ROC-AUC**.
* Key Insight: **Discount Rate** (> 20%) is the single largest driver of order profitability loss.

### 4. 🌐 Real-Time Interactive Web Dashboard
* Flask backend serving dynamic ML inference endpoints.
* Dark-mode glassmorphic single-page frontend powered by ApexCharts, Google Fonts (*Outfit* & *Inter*), and interactive loss prediction risk sliders.

---

## 🏗️ Repository Architecture

```
├── app.py                      # Flask web server & real-time ML API engine
├── ml_pipeline.py              # Modular Python batch training script for all 3 ML models
├── e_commerce.ipynb            # Jupyter Notebook with full EDA, SQL queries, and ML models
├── e commerce sales.sql        # MySQL database schema, transformations & window function queries
├── Sample - Superstore.csv     # Superstore e-commerce transactional dataset
├── templates/
│   └── index.html              # Single-page glassmorphic web dashboard template
├── static/
│   ├── css/style.css           # Glassmorphism dark mode stylesheet
│   └── js/main.js              # Dashboard logic & real-time API integration
├── weekly_sales_forecasting.png
├── customer_segmentation_rfm.png
├── loss_classification_features.png
└── README.md
```

---

## 🚀 Quick Start & Installation

### 1. Clone & Install Dependencies
```bash
git clone <your-repository-url>
cd "E commerce sales analysis"
pip install pandas numpy scikit-learn xgboost flask matplotlib seaborn plotly
```

### 2. Run Machine Learning Pipeline
```bash
python ml_pipeline.py
```

### 3. Launch Interactive Web Dashboard
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

## 📊 SQL Analysis & Database Schema
MySQL schema script `e commerce sales.sql` includes:
* Date string conversion & schema migration (`STR_TO_DATE`).
* Year-over-Year (YoY) growth window functions (`LAG() OVER (ORDER BY year)`).
* Category revenue ranking and profit margin calculations.

---

## 👤 Author
* **Avinash Kumar Pal** ([@palavinashkumar6](mailto:palavinashkumar6@gmail.com))
