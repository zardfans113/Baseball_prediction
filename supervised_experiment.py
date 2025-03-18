import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# 設定日文字體（使用 MS Gothic、Yu Gothic、Meiryo 或 Noto Sans CJK JP）
plt.rcParams['font.sans-serif'] = ['MS Gothic']  # 或 'Meiryo', 'Yu Gothic', 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False  # 避免負號顯示錯誤

# ============ 📌 Step 1: Read and Process Data ============
file_paths = {
    "2021": "2021_qualified_hitters.csv",
    "2022": "2022_qualified_hitters.csv",
    "2023": "2023_qualified_hitters.csv",
    "2024": "2024_qualified_hitters.csv"
}

dfs = []
for year, path in file_paths.items():
    df = pd.read_csv(path)
    df["year"] = int(year)  # Add year column
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# Clean data: Remove unnecessary characters in player names
df_all["選手名"] = df_all["選手名"].str.split(":").str[-1]

# Convert columns to correct data types
float_cols = ["打 率", "出 塁 率", "長 打 率", "O P S"]
int_cols = ["本 塁 打", "打 点"]

for col in float_cols:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
for col in int_cols:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce').fillna(0).astype(int)

# Select features and target variables
features = ["打 席 数", "打 率", "本 塁 打", "出 塁 率", "O P S"]
targets = ["打 率", "本 塁 打", "打 点", "出 塁 率", "長 打 率", "O P S"]

df_cleaned = df_all.dropna(subset=features + targets)

# ============ 📌 Step 2: Split Train/Test Data ============
train_df = df_cleaned[df_cleaned["year"] < 2024]  # Train on 2021-2023
test_df = df_cleaned[df_cleaned["year"] == 2024]  # Test on 2024

X_train = train_df[features]
y_train = train_df[targets]
X_test = test_df[features]
y_test = test_df[targets]

# ============ 📌 Step 3: Experiment - Effect of Training Data Size ============
train_sizes = [0.5, 0.75, 1.0]  # 50%, 75%, 100% training data
results = {}

for size in train_sizes:
    print(f"\n🔹 Training Data Size: {size * 100:.0f}%")
    
    if size == 1.0:
        # Use full training data without splitting
        X_train_partial = X_train
        y_train_partial = y_train
    else:
        # Split only if size < 100%
        X_train_partial, _, y_train_partial, _ = train_test_split(
            X_train, y_train, train_size=size, random_state=42
        )
    
    # Train XGBoost
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=10000, learning_rate=0.001, 
        max_depth=8, subsample=0.9, colsample_bytree=0.7
    )
    xgb_model.fit(X_train_partial, y_train_partial)
    y_pred_xgb = xgb_model.predict(X_test)
    
    # Train RandomForest
    rf_model = RandomForestRegressor(n_estimators=250, max_depth=10, random_state=42)
    rf_model.fit(X_train_partial, y_train_partial)
    y_pred_rf = rf_model.predict(X_test)
    
    # Calculate performance metrics
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    results[size] = {
        "XGBoost MAE": mae_xgb, "RandomForest MAE": mae_rf,
        "XGBoost MSE": mse_xgb, "RandomForest MSE": mse_rf,
        "XGBoost R²": r2_xgb, "RandomForest R²": r2_rf
    }

    print(f"  XGBoost → MAE: {mae_xgb:.4f}, MSE: {mse_xgb:.4f}, R²: {r2_xgb:.4f}")
    print(f"  RandomForest → MAE: {mae_rf:.4f}, MSE: {mse_rf:.4f}, R²: {r2_rf:.4f}")

# ============ 📌 Step 4: Visualizing Results ============
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ["MAE", "MSE", "R²"]
for i, metric in enumerate(metrics):
    axes[i].plot(train_sizes, [results[size][f"XGBoost {metric}"] for size in train_sizes], label="XGBoost", marker="o", linestyle='dashed')
    axes[i].plot(train_sizes, [results[size][f"RandomForest {metric}"] for size in train_sizes], label="RandomForest", marker="o")
    axes[i].set_xlabel("Training Data Size")
    axes[i].set_ylabel(metric)
    axes[i].set_title(f"Effect of Training Data Size on {metric}")
    axes[i].legend()

plt.tight_layout()
plt.show()