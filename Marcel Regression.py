import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 讀取數據集
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

# 划分訓練集（2021-2023）和測試集（2024）
train_df = df_cleaned[df_cleaned["year"] < 2024]
test_df = df_cleaned[df_cleaned["year"] == 2024]

X_train = train_df[features]
X_test = test_df[features]

# 訓練並預測多個目標變數
predictions = {}
mae_scores = {}

df_2025_predictions = pd.DataFrame()
df_2025_predictions["選手名"] = test_df["選手名"]
df_2025_predictions["球 団"] = test_df["球 団"]

for target in targets:
    model = xgb.XGBRegressor(  # 每個變數都應該獨立建立一個新模型
        objective='reg:squarederror',
        n_estimators=10000,
        learning_rate=0.001,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.7
    )
    
    y_train = train_df[target]
    y_test = test_df[target]
    
    model.fit(X_train, y_train)  # 獨立訓練
    y_pred = model.predict(X_test)
    predictions[target] = y_pred
    mae_scores[target] = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error for {target}: {mae_scores[target]}")
    
    # 2025 預測
    df_2025_predictions[f"{target}"] = y_pred

# 確保 "本 塁 打" 和 "打 点" 為整數
for target in ["本 塁 打", "打 点"]:
    df_2025_predictions[target] = df_2025_predictions[target].round().astype(int)

# 儲存預測結果
df_2025_predictions.to_csv("2025_predicted_stats.csv", index=False, encoding='utf-8-sig')
print("Predictions for 2025 saved to 2025_predicted_stats.csv")
