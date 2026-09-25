import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    silhouette_score
)
from xgboost import XGBRegressor, XGBClassifier

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def run_pipeline():
    print("=" * 60)
    print("STARTING E-COMMERCE MACHINE LEARNING PIPELINE")
    print("=" * 60)
    
    # 1. LOAD DATASET
    file_path = "Sample - Superstore.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
        
    df = pd.read_csv(file_path, encoding='latin1')
    print(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Preprocessing Dates
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y', errors='coerce')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%m/%d/%Y', errors='coerce')
    df = df.dropna(subset=['Order Date']).sort_values('Order Date').reset_index(drop=True)
    
    # -------------------------------------------------------------
    # TASK 1: TIME-SERIES WEEKLY SALES FORECASTING (REGRESSION)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("📈 TASK 1: WEEKLY SALES FORECASTING (TIME SERIES REGRESSION)")
    print("=" * 50)
    
    # Aggregate sales by Week
    weekly_sales = df.set_index('Order Date').resample('W-MON')['Sales'].sum().reset_index().rename(columns={'Sales': 'Weekly_Sales'})
    
    # Feature Engineering for Weekly Forecasting
    weekly_sales['Year'] = weekly_sales['Order Date'].dt.year
    weekly_sales['Month'] = weekly_sales['Order Date'].dt.month
    weekly_sales['Week_of_Year'] = weekly_sales['Order Date'].dt.isocalendar().week.astype(int)
    weekly_sales['Quarter'] = weekly_sales['Order Date'].dt.quarter
    
    # Lag and Rolling features
    for lag in [1, 2, 3, 4, 8, 12]:
        weekly_sales[f'Sales_Lag_{lag}'] = weekly_sales['Weekly_Sales'].shift(lag)
        
    for window in [4, 8, 12]:
        weekly_sales[f'Sales_Rolling_Mean_{window}'] = weekly_sales['Weekly_Sales'].shift(1).rolling(window).mean()
        weekly_sales[f'Sales_Rolling_Std_{window}'] = weekly_sales['Weekly_Sales'].shift(1).rolling(window).std()
        
    ml_weekly = weekly_sales.dropna().reset_index(drop=True)
    
    feature_cols_w = [c for c in ml_weekly.columns if c not in ['Order Date', 'Weekly_Sales']]
    X_w = ml_weekly[feature_cols_w]
    y_w = ml_weekly['Weekly_Sales']
    
    split_idx_w = int(len(ml_weekly) * 0.8)
    X_train_w, X_test_w = X_w.iloc[:split_idx_w], X_w.iloc[split_idx_w:]
    y_train_w, y_test_w = y_w.iloc[:split_idx_w], y_w.iloc[split_idx_w:]
    dates_test_w = ml_weekly['Order Date'].iloc[split_idx_w:]
    
    ts_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    }
    
    w_results = []
    w_preds = {}
    
    for name, model in ts_models.items():
        model.fit(X_train_w, y_train_w)
        preds = model.predict(X_test_w)
        w_preds[name] = preds
        
        mae = mean_absolute_error(y_test_w, preds)
        rmse = np.sqrt(mean_squared_error(y_test_w, preds))
        r2 = r2_score(y_test_w, preds)
        
        w_results.append({
            "Model": name,
            "MAE ($)": round(mae, 2),
            "RMSE ($)": round(rmse, 2),
            "R2 Score": round(r2, 4)
        })
        
    w_df = pd.DataFrame(w_results).sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
    print("\n--- Weekly Sales Forecasting Model Evaluation ---")
    print(w_df.to_string(index=False))
    
    # Plot Weekly Forecasting Results
    plt.figure(figsize=(14, 6))
    plt.plot(dates_test_w, y_test_w.values, label='Actual Weekly Sales', color='black', linewidth=2, marker='o')
    best_model_name = w_df.iloc[0]['Model']
    plt.plot(dates_test_w, w_preds[best_model_name], label=f'Forecast ({best_model_name})', color='#1f77b4', linewidth=2, linestyle='--', marker='s')
    plt.title(f"Weekly Sales Forecasting: Actual vs {best_model_name} Forecast", fontsize=14, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Sales ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("weekly_sales_forecasting.png", dpi=300)
    plt.close()
    print("Saved plot: weekly_sales_forecasting.png")
    
    # -------------------------------------------------------------
    # TASK 2: CUSTOMER SEGMENTATION (RFM + K-MEANS CLUSTERING)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("👥 TASK 2: CUSTOMER SEGMENTATION (RFM + K-MEANS)")
    print("=" * 50)
    
    max_date = df['Order Date'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('Customer ID').agg({
        'Order Date': lambda x: (max_date - x.max()).days, # Recency
        'Order ID': 'nunique',                             # Frequency
        'Sales': 'sum'                                     # Monetary
    }).reset_index()
    
    rfm.columns = ['Customer_ID', 'Recency', 'Frequency', 'Monetary']
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    # Silhouette evaluation
    sil_scores = []
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(rfm_scaled)
        sil_scores.append((k, silhouette_score(rfm_scaled, labels)))
        
    best_k = max(sil_scores, key=lambda x: x[1])[0]
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Summarize Clusters
    cluster_summary = rfm.groupby('Cluster').agg({
        'Customer_ID': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).rename(columns={'Customer_ID': 'Customer_Count'}).reset_index()
    
    cluster_summary = cluster_summary.sort_values(by='Monetary', ascending=False).reset_index(drop=True)
    personas = ["Champions / VIPs", "Loyal / High Spend", "Potential Loyalists", "Lapsed / At Risk"][:best_k]
    cluster_summary['Persona'] = personas
    
    cluster_map = dict(zip(cluster_summary['Cluster'], cluster_summary['Persona']))
    rfm['Segment_Persona'] = rfm['Cluster'].map(cluster_map)
    
    print("\n--- RFM Customer Segmentation Summary ---")
    print(cluster_summary.to_string(index=False))
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Segment_Persona', palette='Set2', ax=ax[0], s=60)
    ax[0].set_title("Recency vs Monetary by Persona", fontweight='bold')
    ax[0].set_yscale('log')
    
    sns.scatterplot(data=rfm, x='Frequency', y='Monetary', hue='Segment_Persona', palette='Set2', ax=ax[1], s=60)
    ax[1].set_title("Frequency vs Monetary by Persona", fontweight='bold')
    ax[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig("customer_segmentation_rfm.png", dpi=300)
    plt.close()
    print("Saved plot: customer_segmentation_rfm.png")
    
    # -------------------------------------------------------------
    # TASK 3: ORDER PROFITABILITY / LOSS CLASSIFICATION
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("⚠️ TASK 3: ORDER PROFITABILITY / LOSS CLASSIFICATION")
    print("=" * 50)
    
    df_cls = df.copy()
    df_cls['Is_Loss'] = (df_cls['Profit'] < 0).astype(int)
    
    cat_features = ['Category', 'Sub-Category', 'Segment', 'Region', 'Ship Mode']
    num_features = ['Sales', 'Quantity', 'Discount']
    
    X_cls = df_cls[cat_features + num_features]
    y_cls = df_cls['Is_Loss']
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
        ]
    )
    
    cls_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost Classifier": XGBClassifier(n_estimators=200, learning_rate=0.05, random_state=42, eval_metric='logloss')
    }
    
    cls_results = []
    
    for name, clf in cls_models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
        pipeline.fit(X_train_c, y_train_c)
        preds = pipeline.predict(X_test_c)
        probs = pipeline.predict_proba(X_test_c)[:, 1] if hasattr(pipeline, "predict_proba") else preds
        
        acc = accuracy_score(y_test_c, preds)
        prec = precision_score(y_test_c, preds)
        rec = recall_score(y_test_c, preds)
        f1 = f1_score(y_test_c, preds)
        auc = roc_auc_score(y_test_c, probs)
        
        cls_results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        })
        
    cls_df = pd.DataFrame(cls_results).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    print("\n--- Order Loss Classification Model Performance ---")
    print(cls_df.to_string(index=False))
    
    # Feature Importance for Profit Loss
    rf_cls_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))])
    rf_cls_pipeline.fit(X_train_c, y_train_c)
    
    ohe_cols = list(rf_cls_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_features))
    all_feature_names = num_features + ohe_cols
    importances = rf_cls_pipeline.named_steps['classifier'].feature_importances_
    
    fi_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=fi_df, x='Importance', y='Feature', hue='Feature', legend=False, palette='viridis')
    plt.title("Top Drivers of Unprofitable / Loss Orders (Feature Importance)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("loss_classification_features.png", dpi=300)
    plt.close()
    print("Saved plot: loss_classification_features.png")
    
    print("\n" + "=" * 60)
    print("✅ MACHINE LEARNING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    run_pipeline()
