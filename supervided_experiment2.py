import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import smogn  # smogn v0.1.2
import matplotlib.font_manager as fm

# 設定日文字體（使用 MS Gothic、Yu Gothic、Meiryo 或 Noto Sans CJK JP）
plt.rcParams['font.sans-serif'] = ['MS Gothic']  # 或 'Meiryo', 'Yu Gothic', 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False  # 避免負號顯示錯誤

# ===== Step 1: Read & Process Data =====
file_paths = {
    "2021": "2021_qualified_hitters.csv",
    "2022": "2022_qualified_hitters.csv",
    "2023": "2023_qualified_hitters.csv",
    "2024": "2024_qualified_hitters.csv"
}

dfs = []
for year, path in file_paths.items():
    df = pd.read_csv(path)
    df["year"] = int(year)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# Clean data: remove unnecessary characters from 選手名
df_all["選手名"] = df_all["選手名"].str.split(":").str[-1]

# Convert columns to numeric
float_cols = ["打 率", "出 塁 率", "長 打 率", "O P S"]
int_cols = ["本 塁 打", "打 点"]
for col in float_cols:
    df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
for col in int_cols:
    df_all[col] = pd.to_numeric(df_all[col], errors="coerce").fillna(0).astype(int)

# Define features and targets.
# Note: Some targets also appear in features.
features = ["打 席 数", "打 率", "本 塁 打", "出 塁 率", "O P S"]
targets = ["打 率", "本 塁 打", "打 点", "出 塁 率", "長 打 率", "O P S"]

# Only keep rows with non-missing values for selected features and targets
df_cleaned = df_all.dropna(subset=features + targets)

# ===== Step 2: Split Train/Test Data =====
train_df = df_cleaned[df_cleaned["year"] < 2024]  # 2021-2023
test_df = df_cleaned[df_cleaned["year"] == 2024]    # 2024

X_test = test_df[features]
y_test = test_df[targets]

# ===== Training WITHOUT SMOGN =====
print("\n🚀 Training WITHOUT SMOGN")
results_without_smogn = {}

for target in targets:
    X_train = train_df[features]
    y_train = train_df[target]  # single target at a time

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=10000,
        learning_rate=0.001,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.7
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    # RandomForest
    rf_model = RandomForestRegressor(n_estimators=250, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # Performance metrics
    mae_xgb = mean_absolute_error(y_test[target], y_pred_xgb)
    mse_xgb = mean_squared_error(y_test[target], y_pred_xgb)
    r2_xgb = r2_score(y_test[target], y_pred_xgb)

    mae_rf = mean_absolute_error(y_test[target], y_pred_rf)
    mse_rf = mean_squared_error(y_test[target], y_pred_rf)
    r2_rf = r2_score(y_test[target], y_pred_rf)

    results_without_smogn[target] = {
        "XGBoost MAE": mae_xgb,
        "RandomForest MAE": mae_rf,
        "XGBoost MSE": mse_xgb,
        "RandomForest MSE": mse_rf,
        "XGBoost R²": r2_xgb,
        "RandomForest R²": r2_rf
    }

    print(f"\n🔹 Target: {target}")
    print(f"  XGBoost → MAE: {mae_xgb:.4f}, MSE: {mse_xgb:.4f}, R²: {r2_xgb:.4f}")
    print(f"  RandomForest → MAE: {mae_rf:.4f}, MSE: {mse_rf:.4f}, R²: {r2_rf:.4f}")

# ===== Applying SMOGN for Data Balancing =====
print("\n🚀 Applying SMOGN for Data Balancing")
results_with_smogn = {}

for target in targets:
    print(f"\n🔹 Applying SMOGN to Target: {target}")
    
    # Remove the target column from features for SMOGN (to avoid duplicate columns)
    features_for_smogn = [col for col in features if col != target]
    
    # Create a subset with only the numeric features and the target.
    data_subset = train_df[features_for_smogn + [target]].copy()
    
    # Ensure each column is numeric.
    for col in features_for_smogn + [target]:
        data_subset[col] = pd.to_numeric(data_subset[col], errors='coerce')
    
    # Drop any rows with missing values before applying SMOGN
    data_subset = data_subset.dropna()
    
    try:
        train_df_smogn = smogn.smoter(
            data=data_subset,
            y=target,   # target column name as string
            k=5,        # number of neighbors
            samp_method="balance",
            under_samp=0.5
        )
    except ValueError as e:
        # Catch the error if synthetic data contains missing values.
        print(f"SMOGN failed for target {target} with error: {e}")
        print("Falling back to original training data for this target.")
        train_df_smogn = data_subset.copy()
    
    # Drop any rows with missing values after SMOGN (if any remain)
    train_df_smogn = train_df_smogn.dropna()

    # For training the model, use the features used in SMOGN (which excludes the target).
    X_train_smogn = train_df_smogn[features_for_smogn]
    y_train_smogn = train_df_smogn[target]

    # Train XGBoost (SMOGN)
    xgb_model.fit(X_train_smogn, y_train_smogn)
    # Adjust X_test accordingly (remove target if it was in features)
    X_test_smogn = X_test[features_for_smogn]
    y_pred_xgb_smogn = xgb_model.predict(X_test_smogn)

    # Train RandomForest (SMOGN)
    rf_model.fit(X_train_smogn, y_train_smogn)
    y_pred_rf_smogn = rf_model.predict(X_test_smogn)

    # Performance metrics
    mae_xgb_smogn = mean_absolute_error(y_test[target], y_pred_xgb_smogn)
    mse_xgb_smogn = mean_squared_error(y_test[target], y_pred_xgb_smogn)
    r2_xgb_smogn = r2_score(y_test[target], y_pred_xgb_smogn)

    mae_rf_smogn = mean_absolute_error(y_test[target], y_pred_rf_smogn)
    mse_rf_smogn = mean_squared_error(y_test[target], y_pred_rf_smogn)
    r2_rf_smogn = r2_score(y_test[target], y_pred_rf_smogn)

    results_with_smogn[target] = {
        "XGBoost MAE": mae_xgb_smogn,
        "RandomForest MAE": mae_rf_smogn,
        "XGBoost MSE": mse_xgb_smogn,
        "RandomForest MSE": mse_rf_smogn,
        "XGBoost R²": r2_xgb_smogn,
        "RandomForest R²": r2_rf_smogn
    }

    print(f"\n🔹 Target: {target} (With SMOGN)")
    print(f"  XGBoost → MAE: {mae_xgb_smogn:.4f}, MSE: {mse_xgb_smogn:.4f}, R²: {r2_xgb_smogn:.4f}")
    print(f"  RandomForest → MAE: {mae_rf_smogn:.4f}, MSE: {mse_rf_smogn:.4f}, R²: {r2_rf_smogn:.4f}")

# ===== Visualizing SMOGN Effect =====
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ["MAE", "MSE", "R²"]
for i, metric in enumerate(metrics):
    # XGBoost plots
    axes[i].plot(
        targets,
        [results_without_smogn[t][f"XGBoost {metric}"] for t in targets],
        label="XGBoost (No SMOGN)",
        marker="o", linestyle='dashed'
    )
    axes[i].plot(
        targets,
        [results_with_smogn[t][f"XGBoost {metric}"] for t in targets],
        label="XGBoost (SMOGN)",
        marker="o"
    )
    # RandomForest plots
    axes[i].plot(
        targets,
        [results_without_smogn[t][f"RandomForest {metric}"] for t in targets],
        label="RandomForest (No SMOGN)",
        marker="s", linestyle='dashed'
    )
    axes[i].plot(
        targets,
        [results_with_smogn[t][f"RandomForest {metric}"] for t in targets],
        label="RandomForest (SMOGN)",
        marker="s"
    )

    axes[i].set_xlabel("Target Variables")
    axes[i].set_ylabel(metric)
    axes[i].set_title(f"Effect of SMOGN on {metric}")
    axes[i].legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
