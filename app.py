import os
import sys
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, silhouette_score
from xgboost import XGBRegressor, XGBClassifier

app = Flask(__name__)

# Global cache for dataset and models
DATA_CACHE = {}

def init_ml_engine():
    print("Loading dataset and initializing ML models...")
    file_path = "Sample - Superstore.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    df = pd.read_csv(file_path, encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y', errors='coerce')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%m/%d/%Y', errors='coerce')
    df = df.dropna(subset=['Order Date']).sort_values('Order Date').reset_index(drop=True)
    DATA_CACHE['df'] = df

    # 1. KPI Aggregates
    total_sales = float(df['Sales'].sum())
    total_profit = float(df['Profit'].sum())
    total_orders = int(df['Order ID'].nunique())
    total_customers = int(df['Customer ID'].nunique())
    profit_margin = round((total_profit / total_sales) * 100, 2)
    loss_orders_pct = round((df['Profit'] < 0).mean() * 100, 2)

    DATA_CACHE['kpis'] = {
        'total_sales': round(total_sales, 2),
        'total_profit': round(total_profit, 2),
        'total_orders': total_orders,
        'total_customers': total_customers,
        'profit_margin': profit_margin,
        'loss_orders_pct': loss_orders_pct
    }

    # 2. Time-Series Weekly Sales Forecasting Model
    weekly_sales = df.set_index('Order Date').resample('W-MON')['Sales'].sum().reset_index().rename(columns={'Sales': 'Weekly_Sales'})
    weekly_sales['Year'] = weekly_sales['Order Date'].dt.year
    weekly_sales['Month'] = weekly_sales['Order Date'].dt.month
    weekly_sales['Week_of_Year'] = weekly_sales['Order Date'].dt.isocalendar().week.astype(int)
    weekly_sales['Quarter'] = weekly_sales['Order Date'].dt.quarter

    for lag in [1, 2, 3, 4, 8, 12]:
        weekly_sales[f'Sales_Lag_{lag}'] = weekly_sales['Weekly_Sales'].shift(lag)
    for window in [4, 8, 12]:
        weekly_sales[f'Sales_Rolling_Mean_{window}'] = weekly_sales['Weekly_Sales'].shift(1).rolling(window).mean()
        weekly_sales[f'Sales_Rolling_Std_{window}'] = weekly_sales['Weekly_Sales'].shift(1).rolling(window).std()

    ml_weekly = weekly_sales.dropna().reset_index(drop=True)
    features_w = [c for c in ml_weekly.columns if c not in ['Order Date', 'Weekly_Sales']]
    X_w = ml_weekly[features_w]
    y_w = ml_weekly['Weekly_Sales']

    split_idx_w = int(len(ml_weekly) * 0.8)
    X_train_w, X_test_w = X_w.iloc[:split_idx_w], X_w.iloc[split_idx_w:]
    y_train_w, y_test_w = y_w.iloc[:split_idx_w], y_w.iloc[split_idx_w:]
    dates_test_w = ml_weekly['Order Date'].iloc[split_idx_w:].dt.strftime('%Y-%m-%d').tolist()

    ts_models = {
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    }

    ts_eval = []
    ts_preds = {}
    for name, model in ts_models.items():
        model.fit(X_train_w, y_train_w)
        pred = model.predict(X_test_w)
        ts_preds[name] = [round(float(p), 2) for p in pred]
        ts_eval.append({
            'model': name,
            'mae': round(float(mean_absolute_error(y_test_w, pred)), 2),
            'rmse': round(float(np.sqrt(mean_squared_error(y_test_w, pred))), 2),
            'r2': round(float(r2_score(y_test_w, pred)), 4)
        })

    DATA_CACHE['forecasting'] = {
        'dates': dates_test_w,
        'actual': [round(float(a), 2) for a in y_test_w.values],
        'predictions': ts_preds,
        'metrics': sorted(ts_eval, key=lambda x: x['r2'], reverse=True)
    }

    # 3. RFM Customer Segmentation
    max_date = df['Order Date'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('Customer ID').agg({
        'Order Date': lambda x: (max_date - x.max()).days,
        'Order ID': 'nunique',
        'Sales': 'sum'
    }).reset_index().rename(columns={'Order Date': 'Recency', 'Order ID': 'Frequency', 'Sales': 'Monetary'})

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    cluster_summary = rfm.groupby('Cluster').agg({
        'Customer ID': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).rename(columns={'Customer ID': 'Customer_Count'}).reset_index()

    cluster_summary = cluster_summary.sort_values(by='Monetary', ascending=False).reset_index(drop=True)
    cluster_summary['Persona'] = ['Champions / VIPs', 'Loyal / High Spend', 'Lapsed / At Risk']
    cluster_map = dict(zip(cluster_summary['Cluster'], cluster_summary['Persona']))
    rfm['Persona'] = rfm['Cluster'].map(cluster_map)

    DATA_CACHE['rfm'] = {
        'summary': cluster_summary.to_dict(orient='records'),
        'scatter': rfm[['Customer ID', 'Recency', 'Frequency', 'Monetary', 'Persona']].head(250).to_dict(orient='records')
    }

    # 4. Order Profit Loss Classifier Pipeline
    df_cls = df.copy()
    df_cls['Is_Loss'] = (df_cls['Profit'] < 0).astype(int)

    cat_cols = ['Category', 'Sub-Category', 'Segment', 'Region', 'Ship Mode']
    num_cols = ['Sales', 'Quantity', 'Discount']

    X_c = df_cls[cat_cols + num_cols]
    y_c = df_cls['Is_Loss']

    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42, stratify=y_c)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
    ])

    xgb_clf = XGBClassifier(n_estimators=200, learning_rate=0.05, random_state=42, eval_metric='logloss')
    loss_pipeline = Pipeline([('preprocessor', preprocessor), ('clf', xgb_clf)])
    loss_pipeline.fit(X_tr_c, y_tr_c)
    preds_c = loss_pipeline.predict(X_te_c)
    probs_c = loss_pipeline.predict_proba(X_te_c)[:, 1]

    DATA_CACHE['loss_pipeline'] = loss_pipeline
    DATA_CACHE['loss_metrics'] = {
        'model': 'XGBoost Classifier',
        'accuracy': round(float(accuracy_score(y_te_c, preds_c)), 4),
        'precision': round(float(precision_score(y_te_c, preds_c)), 4),
        'recall': round(float(recall_score(y_te_c, preds_c)), 4),
        'f1': round(float(f1_score(y_te_c, preds_c)), 4),
        'roc_auc': round(float(roc_auc_score(y_te_c, probs_c)), 4)
    }

    # Feature Importance for Loss Classification
    rf_pipe = Pipeline([('preprocessor', preprocessor), ('clf', RandomForestClassifier(n_estimators=200, random_state=42))])
    rf_pipe.fit(X_tr_c, y_tr_c)
    ohe_names = list(rf_pipe.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols))
    all_feat_names = num_cols + ohe_names
    importances = rf_pipe.named_steps['clf'].feature_importances_
    fi_df = pd.DataFrame({'feature': all_feat_names, 'importance': importances}).sort_values(by='importance', ascending=False).head(10)
    DATA_CACHE['feature_importance'] = fi_df.to_dict(orient='records')

    # Category / Regional Summaries
    cat_sales = df.groupby('Category').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index().to_dict(orient='records')
    subcat_sales = df.groupby('Sub-Category').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index().sort_values('Sales', ascending=False).head(10).to_dict(orient='records')
    region_sales = df.groupby('Region').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index().to_dict(orient='records')

    DATA_CACHE['eda'] = {
        'category': cat_sales,
        'sub_category': subcat_sales,
        'region': region_sales
    }
    print("ML Engine initialization complete!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/dashboard_data')
def get_dashboard_data():
    return jsonify({
        'kpis': DATA_CACHE['kpis'],
        'forecasting': DATA_CACHE['forecasting'],
        'rfm': DATA_CACHE['rfm'],
        'loss_metrics': DATA_CACHE['loss_metrics'],
        'feature_importance': DATA_CACHE['feature_importance'],
        'eda': DATA_CACHE['eda']
    })

@app.route('/api/predict_loss', methods=['POST'])
def predict_loss():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([{
            'Category': data.get('category', 'Furniture'),
            'Sub-Category': data.get('sub_category', 'Tables'),
            'Segment': data.get('segment', 'Consumer'),
            'Region': data.get('region', 'Central'),
            'Ship Mode': data.get('ship_mode', 'Standard Class'),
            'Sales': float(data.get('sales', 250.0)),
            'Quantity': int(data.get('quantity', 3)),
            'Discount': float(data.get('discount', 0.40))
        }])

        pipeline = DATA_CACHE['loss_pipeline']
        prob_loss = float(pipeline.predict_proba(input_df)[0][1])
        is_loss = bool(prob_loss >= 0.5)

        risk_level = "High Loss Risk 🚨" if prob_loss > 0.65 else ("Moderate Risk ⚠️" if prob_loss >= 0.35 else "Profitable / Low Risk ✅")

        return jsonify({
            'status': 'success',
            'is_loss': is_loss,
            'loss_probability': round(prob_loss * 100, 2),
            'risk_level': risk_level,
            'recommendation': "Reduce discount below 20% to avoid loss!" if prob_loss >= 0.35 else "Order margins are healthy."
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    init_ml_engine()
    app.run(host='127.0.0.1', port=5000, debug=False)
