import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 設定日文字體（使用 MS Gothic、Yu Gothic、Meiryo 或 Noto Sans CJK JP）
plt.rcParams['font.sans-serif'] = ['MS Gothic']  # 或 'Meiryo', 'Yu Gothic', 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False  # 避免負號顯示錯誤


# ============ 📌 Step 1: 讀取並整理數據 ============

file_paths = {
    "2021": "2021_qualified_hitters.csv",
    "2022": "2022_qualified_hitters.csv",
    "2023": "2023_qualified_hitters.csv",
    "2024": "2024_qualified_hitters.csv"
}

dfs = []
for year, path in file_paths.items():
    df = pd.read_csv(path)
    df["year"] = int(year)  # 添加年份標記
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# 清理數據：移除選手名前的數字標記
df_all["選手名"] = df_all["選手名"].str.split(":").str[-1]

# 轉換數據類型
float_cols = ["打 率", "出 塁 率", "長 打 率", "O P S"]
int_cols = ["本 塁 打", "打 点"]

for col in float_cols:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
for col in int_cols:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce').fillna(0).astype(int)  # 確保為整數

# 選擇特徵與目標變數
features = ["打 席 数", "打 率", "本 塁 打", "出 塁 率", "O P S"]
targets = ["打 率", "本 塁 打", "打 点", "出 塁 率", "長 打 率", "O P S"]

df_cleaned = df_all.dropna(subset=features + targets)

# ============ 📌 Step 2: 訓練 XGBoost 和隨機森林來預測 2024 ============

# 使用 2021-2023 訓練，2024 測試
train_df = df_cleaned[df_cleaned["year"] < 2024]
test_df = df_cleaned[df_cleaned["year"] == 2024]

X_train = train_df[features]
X_test = test_df[features]

# 訓練並預測多個目標變數
predictions_xgb = {}
predictions_rf = {}
mae_scores_xgb = {}
mae_scores_rf = {}
mse_scores_xgb = {}
mse_scores_rf = {}
r2_scores_xgb = {}
r2_scores_rf = {}

df_2024_predictions_xgb = pd.DataFrame()
df_2024_predictions_rf = pd.DataFrame()

df_2024_predictions_xgb["選手名"] = test_df["選手名"]
df_2024_predictions_xgb["球 団"] = test_df["球 団"]
df_2024_predictions_rf["選手名"] = test_df["選手名"]
df_2024_predictions_rf["球 団"] = test_df["球 団"]

for target in targets:
    # **XGBoost 模型**
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=10000,
        learning_rate=0.001,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.7
    )
    y_train = train_df[target]
    y_test = test_df[target]
    
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    
    predictions_xgb[target] = y_pred_xgb
    mae_scores_xgb[target] = mean_absolute_error(y_test, y_pred_xgb)
    mse_scores_xgb[target] = mean_squared_error(y_test, y_pred_xgb)
    r2_scores_xgb[target] = r2_score(y_test, y_pred_xgb)
    
    df_2024_predictions_xgb[f"{target}"] = y_pred_xgb
    
    # **隨機森林模型**
    rf_model = RandomForestRegressor(
        n_estimators=250,  
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    predictions_rf[target] = y_pred_rf
    mae_scores_rf[target] = mean_absolute_error(y_test, y_pred_rf)
    mse_scores_rf[target] = mean_squared_error(y_test, y_pred_rf)
    r2_scores_rf[target] = r2_score(y_test, y_pred_rf)
    
    df_2024_predictions_rf[f"{target}"] = y_pred_rf

    print(f"🔹 {target} - XGBoost MAE: {mae_scores_xgb[target]:.4f} | Random Forest MAE: {mae_scores_rf[target]:.4f}")
    print(f"   ↳ XGBoost MSE: {mse_scores_xgb[target]:.4f} | Random Forest MSE: {mse_scores_rf[target]:.4f}")
    print(f"   ↳ XGBoost R²: {r2_scores_xgb[target]:.4f} | Random Forest R²: {r2_scores_rf[target]:.4f}")

# ============ 📌 Step 3: 視覺化比較 ============

plt.figure(figsize=(10, 6))
x_labels = targets
x = np.arange(len(x_labels))

plt.bar(x - 0.2, mae_scores_xgb.values(), width=0.4, label="XGBoost", color="blue")
plt.bar(x + 0.2, mae_scores_rf.values(), width=0.4, label="Random Forest", color="green")
plt.xticks(x, x_labels, rotation=45)
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("XGBoost vs. Random Forest MAE 比較")
plt.legend()
plt.show()

# **誤差分佈散點圖**
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, target in enumerate(targets):
    ax = axes[i]
    ax.scatter(test_df[target], predictions_xgb[target], label="XGBoost", alpha=0.6, color="blue")
    ax.scatter(test_df[target], predictions_rf[target], label="Random Forest", alpha=0.6, color="green")
    ax.plot([min(test_df[target]), max(test_df[target])], [min(test_df[target]), max(test_df[target])], color="red", linestyle="dashed")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(target)
    ax.legend()

plt.tight_layout()
plt.show()
# ============ 📌 Step 3: 使用 2021-2024 訓練並預測 2025 ============

train_df_2025 = df_cleaned[df_cleaned["year"] <= 2024]  # **使用 2021-2024 訓練**
X_train_2025 = train_df_2025[features]

# **🚀 調整 2021-2024 的數據權重**
train_df_2025["weight"] = train_df_2025["year"].map({
    2021: 0.5,  
    2022: 1.0,
    2023: 1.5,  
    2024: 2.0
})

df_2025_predictions_xgb = pd.DataFrame()
df_2025_predictions_rf = pd.DataFrame()

df_2025_predictions_xgb["選手名"] = df_2024_predictions_xgb["選手名"]
df_2025_predictions_xgb["球 団"] = df_2024_predictions_xgb["球 団"]
df_2025_predictions_rf["選手名"] = df_2024_predictions_rf["選手名"]
df_2025_predictions_rf["球 団"] = df_2024_predictions_rf["球 団"]

# **創建 2025 測試數據**
X_test_2025 = X_test.copy()
X_test_2025 += X_test_2025 * np.random.uniform(-0.05, 0.05, X_test_2025.shape)  # **模擬 2025 的變化**

for target in targets:
    y_train_2025 = train_df_2025[target]

    # **XGBoost**
    xgb_model.fit(X_train_2025, y_train_2025, sample_weight=train_df_2025["weight"])
    df_2025_predictions_xgb[target] = xgb_model.predict(X_test_2025)

    # **隨機森林**
    rf_model.fit(X_train_2025, y_train_2025, sample_weight=train_df_2025["weight"])
    df_2025_predictions_rf[target] = rf_model.predict(X_test_2025)

# 確保 "本塁打" 和 "打点" 為整數
for target in ["本 塁 打", "打 点"]:
    df_2025_predictions_xgb[target] = df_2025_predictions_xgb[target].round().astype(int)
    df_2025_predictions_rf[target] = df_2025_predictions_rf[target].round().astype(int)

# 儲存預測結果
df_2025_predictions_xgb.to_csv("2025_predicted_stats_xgb.csv", index=False, encoding='utf-8-sig')
df_2025_predictions_rf.to_csv("2025_predicted_stats_rf.csv", index=False, encoding='utf-8-sig')

print("✅ 2025 年預測已完成，結果存為 2025_predicted_stats_xgb.csv 和 2025_predicted_stats_rf.csv")